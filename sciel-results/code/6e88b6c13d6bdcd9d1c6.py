import numpy as np


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    n = len(inventory)
    q = inventory.astype(np.float64)
    
    # Base risk-neutral depth
    d0 = 1.0 / k
    
    # Risk adjustment: scale with inventory and risk parameters
    # The optimal adjustment is proportional to gamma * sigma^2 * q / (A * k)
    # but we need to be careful about the time horizon
    
    # Time remaining affects how much we care about risk
    t_remaining = T - t
    # As t_remaining -> 0, we don't care about risk (no future penalty)
    # As t_remaining -> T, we care more
    
    # The risk penalty accumulates over time, so the effective risk aversion
    # is proportional to t_remaining
    risk_scale = gamma * sigma**2 * t_remaining / (2.0 * A * k * dt)
    
    # But we need to clamp to avoid extreme values
    # The maximum inventory is Q, so the adjustment should be bounded
    max_adj = 3.0 / k  # Don't go beyond the validity limit
    
    # Linear adjustment: d_bid = d0 + c*q for q>0, d_ask = d0 - c*q for q<0
    # Actually, for q>0: increase d_bid (worse bid), decrease d_ask (better ask)
    # For q<0: decrease d_bid (better bid), increase d_ask (worse ask)
    
    # c = risk_scale, but clamp
    c = np.clip(risk_scale, 0.0, max_adj / max(Q, 1.0))
    
    # Bid depth: if q > 0, make it deeper (worse); if q < 0, make it shallower (better)
    # d_bid = d0 + c * q (but clamp to valid range)
    d_bid = d0 + c * q
    # Ask depth: if q > 0, make it shallower (better); if q < 0, make it deeper (worse)
    d_ask = d0 - c * q
    
    # Clamp to valid range: must be > -3/k and finite
    min_depth = -3.0 / k + 1e-9
    d_bid = np.maximum(d_bid, min_depth)
    d_ask = np.maximum(d_ask, min_depth)
    
    # Ensure non-negative (depths should be positive distances)
    d_bid = np.maximum(d_bid, 0.0)
    d_ask = np.maximum(d_ask, 0.0)
    
    # Handle boundary: no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q, set bid to inf (no bid quote)
    # At q=-Q, set ask to inf (no ask quote)
    q_int = inventory.astype(np.int64)
    d_bid = np.where(q_int >= Q, np.inf, d_bid)
    d_ask = np.where(q_int <= -Q, np.inf, d_ask)
    
    return d_bid, d_ask
