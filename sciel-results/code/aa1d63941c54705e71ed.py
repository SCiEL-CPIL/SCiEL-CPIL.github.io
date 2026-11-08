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
    t_rem = T - t
    
    # Base depth calculation
    # Heuristic: Balance arrival rate with risk
    # d ~ (1/k) ln( A dt / Risk )
    # Risk scale ~ gamma * sigma^2 * t_rem
    risk_scale = gamma * sigma**2 * t_rem
    if risk_scale < 1e-9:
        risk_scale = 1e-9
        
    # A dt is the expected number of arrivals at d=0 (roughly, if k*d is small)
    # Actually p = 1 - exp(-A exp(-kd) dt).
    # At d=0, p = 1 - exp(-A dt).
    # We want d such that the marginal utility matches risk.
    # Let's use a simple log form.
    # If A dt is large, d should be small (good price).
    # If risk is high, d should be large (bad price).
    
    # d_base = (1/k) * ln( (A * dt) / (risk_scale + epsilon) )
    # This might be negative.
    # Let's clamp d_base to a minimum of 0.
    
    d_base = (1.0 / k) * np.log( (A * dt) / (risk_scale + 1e-6) )
    if d_base < 0:
        d_base = 0.0
        
    # Inventory shift
    # d_bid = d_base + C * q
    # d_ask = d_base - C * q
    # C is the sensitivity to inventory.
    # From AVS, shift is gamma * sigma^2 * (T-t) * q / 2.
    # Here, since we earn depth, the incentive to widen is stronger?
    # Let's use C = gamma * sigma^2 * t_rem / (2 * k) ? No, units of C must be 1 (depth per unit inventory).
    # gamma [1/Time?], sigma [Price/sqrt(Time)], t_rem [Time].
    # gamma * sigma^2 * t_rem has units of Price.
    # Depth is Price.
    # So C = gamma * sigma^2 * t_rem / 2.
    
    C = 0.5 * gamma * sigma**2 * t_rem
    
    # Vectorized calculation
    q = np.asarray(inventory)
    
    d_bid = d_base + C * q
    d_ask = d_base - C * q
    
    # Clamp to 0
    d_bid = np.maximum(d_bid, 0.0)
    d_ask = np.maximum(d_ask, 0.0)
    
    # Boundary conditions
    # If q == Q, no bid fill allowed -> d_bid = inf
    # If q == -Q, no ask fill allowed -> d_ask = inf
    d_bid = np.where(q == Q, np.inf, d_bid)
    d_ask = np.where(q == -Q, np.inf, d_ask)
    
    # Ensure float64
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    return d_bid, d_ask
