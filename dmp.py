import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Market Expansion Forecast Dashboard",
    layout="wide"
)

# -----------------------------
# Demo market dataset
# -----------------------------
data = pd.DataFrame({
    "Market": ["UK", "Germany", "France", "Spain", "Italy", "Netherlands", "Poland"],
    "Search Volume": [90000, 85000, 62000, 58000, 52000, 35000, 47000],
    "CPC": [2.80, 2.60, 2.30, 1.60, 1.70, 2.10, 1.20],
    "Competition Index": [85, 80, 72, 60, 65, 68, 48],
    "Intent Score": [82, 78, 68, 76, 70, 73, 79],
    "Trend Growth": [4, 6, 5, 13, 8, 7, 18],
    "GDP per Capita Index": [82, 88, 80, 70, 68, 86, 58],
    "Ecommerce Maturity": [88, 84, 76, 72, 68, 82, 66],
    "AI Discoverability": [78, 72, 66, 70, 64, 75, 69],
})

# Demo target for regression
data["Revenue Potential"] = (
    data["Search Volume"] * 0.00035
    + data["Intent Score"] * 0.9
    + data["Trend Growth"] * 1.8
    + data["GDP per Capita Index"] * 0.35
    + data["Ecommerce Maturity"] * 0.45
    + data["AI Discoverability"] * 0.30
    - data["CPC"] * 8
    - data["Competition Index"] * 0.45
)

features = [
    "Search Volume",
    "CPC",
    "Competition Index",
    "Intent Score",
    "Trend Growth",
    "GDP per Capita Index",
    "Ecommerce Maturity",
    "AI Discoverability",
]

X = data[features]
y = data["Revenue Potential"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

data["Predicted Revenue Score"] = model.predict(X_scaled).round(1)

min_score = data["Predicted Revenue Score"].min()
max_score = data["Predicted Revenue Score"].max()

data["Expansion Score"] = (
    (data["Predicted Revenue Score"] - min_score)
    / (max_score - min_score)
    * 100
).round(0)

# -----------------------------
# Forecast function
# -----------------------------
def forecast_market(budget, cpc, conversion_rate, value_per_conversion):
    clicks = int(budget / cpc) if cpc > 0 else 0
    conversions = clicks * (conversion_rate / 100)
    revenue = conversions * value_per_conversion
    cpa = budget / conversions if conversions > 0 else float("inf")
    roas = revenue / budget if budget > 0 else 0
    profit = revenue - budget

    return {
        "clicks": clicks,
        "conversions": conversions,
        "revenue": revenue,
        "cpa": cpa,
        "roas": roas,
        "profit": profit,
    }

# -----------------------------
# App
# -----------------------------
st.title("Market Expansion Forecast Dashboard")

st.caption(
    "Interactive forecasting tool to estimate clicks, conversions, revenue, CPA and ROAS by market."
)

st.info(
    "This is a demo model using anonymised/sample data. In a real project, the model would use updated market data "
    "from tools such as Google Trends, Google Keyword Planner, SEMrush, Similarweb and public economic sources, "
    "combined with historical client-side data such as Google Analytics, campaign performance, conversion rates, CPCs and revenue data."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Market & Budget Inputs")

selected_market = st.sidebar.selectbox(
    "Select market",
    data["Market"].tolist()
)

market_row = data[data["Market"] == selected_market].iloc[0]

budget = st.sidebar.number_input(
    "Budget (£)",
    min_value=0.0,
    value=5000.0,
    step=500.0
)

cpc = st.sidebar.number_input(
    "Average CPC (£)",
    min_value=0.01,
    value=float(market_row["CPC"]),
    step=0.10
)

conversion_rate = st.sidebar.number_input(
    "Expected Conversion Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.25
)

value_per_conversion = st.sidebar.number_input(
    "Value per Conversion (£)",
    min_value=0.0,
    value=120.0,
    step=10.0
)

st.sidebar.header("Market Signal Inputs")

search_volume = st.sidebar.number_input(
    "Search Volume",
    min_value=0,
    value=int(market_row["Search Volume"]),
    step=1000
)

competition = st.sidebar.slider(
    "Competition Index",
    0,
    100,
    int(market_row["Competition Index"])
)

intent = st.sidebar.slider(
    "Purchase Intent Score",
    0,
    100,
    int(market_row["Intent Score"])
)

trend_growth = st.sidebar.number_input(
    "Trend Growth (%)",
    value=float(market_row["Trend Growth"]),
    step=1.0
)

gdp_index = st.sidebar.slider(
    "GDP per Capita Index",
    0,
    100,
    int(market_row["GDP per Capita Index"])
)

ecommerce = st.sidebar.slider(
    "Ecommerce Maturity",
    0,
    100,
    int(market_row["Ecommerce Maturity"])
)

ai_discoverability = st.sidebar.slider(
    "AI Discoverability",
    0,
    100,
    int(market_row["AI Discoverability"])
)

# -----------------------------
# Custom prediction
# -----------------------------
custom_input = pd.DataFrame([{
    "Search Volume": search_volume,
    "CPC": cpc,
    "Competition Index": competition,
    "Intent Score": intent,
    "Trend Growth": trend_growth,
    "GDP per Capita Index": gdp_index,
    "Ecommerce Maturity": ecommerce,
    "AI Discoverability": ai_discoverability,
}])

custom_scaled = scaler.transform(custom_input)
predicted_revenue_score = model.predict(custom_scaled)[0]

forecast = forecast_market(
    budget,
    cpc,
    conversion_rate,
    value_per_conversion
)

# -----------------------------
# Main KPIs
# -----------------------------
st.header(f"{selected_market} Forecast")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Budget", f"£{budget:,.0f}")
c2.metric("Estimated Clicks", f"{forecast['clicks']:,}")
c3.metric("Estimated Conversions", f"{forecast['conversions']:.1f}")
c4.metric("Predicted Revenue Score", f"{predicted_revenue_score:.1f}")

c5, c6, c7, c8 = st.columns(4)

c5.metric("Revenue", f"£{forecast['revenue']:,.0f}")
c6.metric("CPA", "∞" if math.isinf(forecast["cpa"]) else f"£{forecast['cpa']:.2f}")
c7.metric("ROAS", f"{forecast['roas']:.2f}")
c8.metric("Profit", f"£{forecast['profit']:,.0f}")

# -----------------------------
# Recommendation
# -----------------------------
st.header("Strategic Interpretation")

insights = []

if forecast["roas"] >= 3:
    insights.append(f"**{selected_market}** looks commercially attractive based on the current budget and conversion assumptions.")
elif forecast["roas"] >= 1.5:
    insights.append(f"**{selected_market}** shows potential, but performance would depend on optimisation and market validation.")
else:
    insights.append(f"**{selected_market}** may be risky under the current assumptions. Budget, CPC or conversion rate would need improvement.")

if cpc < data["CPC"].mean():
    insights.append("- CPC is below the benchmark market average, suggesting more efficient traffic acquisition.")
else:
    insights.append("- CPC is above the benchmark market average, so acquisition efficiency may be more challenging.")

if trend_growth > data["Trend Growth"].mean():
    insights.append("- Search trend growth is above average, suggesting positive demand momentum.")
else:
    insights.append("- Search trend growth is below average, suggesting slower market momentum.")

if competition < data["Competition Index"].mean():
    insights.append("- Competition intensity is below average, which may make market entry easier.")
else:
    insights.append("- Competition intensity is above average, so differentiation and strong execution would be important.")

st.success("\n\n".join(insights))

# -----------------------------
# Benchmark table
# -----------------------------
st.header("Benchmark Market Ranking")

ranking = data.sort_values("Expansion Score", ascending=False)[[
    "Market",
    "Expansion Score",
    "Predicted Revenue Score",
    "Search Volume",
    "CPC",
    "Competition Index",
    "Intent Score",
    "Trend Growth",
    "AI Discoverability",
]]

st.dataframe(ranking, use_container_width=True, hide_index=True)

# -----------------------------
# Forecast comparison across markets
# -----------------------------
st.header("Budget Forecast Across Markets")

comparison_rows = []

for _, row in data.iterrows():
    result = forecast_market(
        budget,
        row["CPC"],
        conversion_rate,
        value_per_conversion
    )

    comparison_rows.append({
        "Market": row["Market"],
        "Budget (£)": budget,
        "CPC (£)": row["CPC"],
        "Estimated Clicks": result["clicks"],
        "Estimated Conversions": round(result["conversions"], 1),
        "Revenue (£)": round(result["revenue"], 0),
        "CPA (£)": round(result["cpa"], 2),
        "ROAS": round(result["roas"], 2),
        "Profit (£)": round(result["profit"], 0),
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values("ROAS", ascending=False)

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# -----------------------------
# Charts
# -----------------------------
st.header("Visuals")

fig1 = plt.figure()
plt.bar(comparison_df["Market"], comparison_df["Estimated Conversions"])
plt.xticks(rotation=45)
plt.ylabel("Estimated Conversions")
plt.title("Estimated Conversions by Market")
st.pyplot(fig1)

fig2 = plt.figure()
plt.bar(comparison_df["Market"], comparison_df["ROAS"])
plt.xticks(rotation=45)
plt.ylabel("ROAS")
plt.title("Estimated ROAS by Market")
st.pyplot(fig2)

# -----------------------------
# Methodology
# -----------------------------
st.header("Methodology")

st.write(
    """
    This dashboard combines two layers:

    **1. Market expansion prediction**  
    A linear regression model estimates market revenue potential using search demand, CPC, competition,
    purchase intent, growth, economic maturity, ecommerce maturity and AI discoverability.

    **2. Budget forecast simulation**  
    The dashboard estimates clicks, conversions, revenue, CPA, ROAS and profit from user-defined budget,
    CPC, conversion rate and value-per-conversion assumptions.

    The model is designed as a decision-support tool for international expansion planning.
    """
)

st.header("Limitations")

st.write(
    """
    - Current data is synthetic/sample data for portfolio demonstration.
    - Real-world use would require live market data and historical performance validation.
    - Forecasts depend heavily on conversion rate, CPC, brand awareness, landing page quality and local execution.
    - The tool should support strategic prioritisation, not replace commercial judgement.
    """
)