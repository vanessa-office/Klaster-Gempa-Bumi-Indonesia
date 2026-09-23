import joblib
import pandas as pd
import streamlit as st

st.title('Klaster Gempa Bumi Indonesia')
st.write('Masukkan data sebuah gempa untuk mengetahui klasternya (model K-Means).')

scaler = joblib.load('scaler_gempa.joblib')
km = joblib.load('kmeans_gempa.joblib')

lat = st.number_input('Lintang (negatif = LS)', -11.0, 6.0, -2.0)
lon = st.number_input('Bujur (BT)', 94.0, 142.0, 118.0)
depth = st.number_input('Kedalaman (km)', 0.0, 700.0, 30.0)
mag = st.number_input('Magnitudo', 1.0, 9.5, 4.5)

data = pd.DataFrame([[lat, lon, depth, mag]],
                    columns=['latitude', 'longitude', 'depth', 'magnitude'])
k = km.predict(scaler.transform(data))[0]

st.success(f'Gempa ini masuk ke Klaster {k}')
st.write('Profil setiap klaster dapat dilihat pada notebook (bagian Evaluation).')
