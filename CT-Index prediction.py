# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:21:43 2025

@author: Xiao
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def train_rf_model(file_path):

    df = pd.read_excel(file_path, sheet_name='CT')
    

    features = [
        'Additives', 'Design Gyration', 'RAP', 'BSG (field)', 'Air Voids (field)',
        'VMA (field)', 'Dust/Binder (Field)', 'Ignition Oven AC (%) (Field)', 'Slip AC Content (%) (Field)'
    ]
    target = 'Avg. CTindex'
    
  
    existing_features = [f for f in features if f in df.columns]
    df_model = df[[target] + existing_features].dropna()
    X = df_model[existing_features]
    y = df_model[target]


    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, existing_features, df_model


st.title("CT-Index Prediction Web Tool")


model, features, df_model = train_rf_model("Result Summary.xlsx")

st.header("Input Features")
input_data = {}
for f in features:
    input_data[f] = st.number_input(f, value=float(df_model[f].mean()))
input_df = pd.DataFrame([input_data])


prediction = model.predict(input_df)[0]
st.subheader(f"Predicted CT-Index: {prediction:.2f}")


st.subheader("Model Training Fit: Predicted vs Actual")
X_all = df_model[features]
y_all = df_model['Avg. CTindex']
y_pred_all = model.predict(X_all)

fig, ax = plt.subplots()
sns.scatterplot(x=y_all, y=y_pred_all, ax=ax)
ax.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'r--')
ax.set_xlabel("True Avg. CTindex")
ax.set_ylabel("Predicted Avg. CTindex")
ax.set_title("Random Forest Prediction vs Actual (Training Set)")
st.pyplot(fig)
