"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Cache for pre-computed optimal depths
# Key: (sigma, A, k, gamma, Q, steps)
# Value: dict with 'bid' and 'ask' arrays of shape (steps, 2*Q+1)
_OPT_TABLES = {}

def _solve_optimal_depths_vectorized(A, k, dt, deltas):
    """
    Approximate solution for optimal depths using first-order expansion.
    x* approx (1/k) * (1 - delta)
    Valid if x* > 0 and J(x*) > 0.
    """
    n = len(deltas)
    best_x = np.full(n, np.inf, dtype=np.float64)
    best_val = np.zeros(n, dtype=np.float64)
    
    # Parameters
    lam = A * dt
    if lam <= 0:
        return best_x, best_val
        
    # Approximation: y* = lam * exp(delta - 1)
    # x* = (1/k) * (1 - delta)
    x_approx = (1.0 / k) * (1.0 - deltas)
    
    # Only consider positive depths
    valid_mask = x_approx > 0
    
    # Calculate probability and value for valid ones
    # y = lam * exp(-k*x) = lam * exp(-(1-delta)) = lam * exp(delta - 1)
    # prob = 1 - exp(-y)
    # val = prob * (x + delta)
    
    # To avoid overflow/underflow in exp for extreme deltas, clip y
    y_approx = lam * np.exp(np.clip(deltas[valid_mask] - 1.0, -50, 50))
    prob = 1.0 - np.exp(-y_approx)
    val_approx = prob * (x_approx[valid_mask] + deltas[valid_mask])
    
    # If val <= 0, no quote
    quote_mask = val_approx > 0
    
    # Assign results
    best_x[valid_mask] = np.where(quote_mask, x_approx[valid_mask], np.inf)
    best_val[valid_mask] = np.where(quote_mask, val_approx, 0.0)
    
    return best_x, best_val

def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    Uses fast approximation for depth calculation.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Initialize value function
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    # Pre-compute costs
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5.0:
            break
            
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        delta_b = np.zeros(2*Q + 1, dtype=np.float64)
        delta_a = np.zeros(2*Q + 1, dtype=np.float64)
        
        # Valid bids: q < Q => indices 0 to 2Q-1
        delta_b[:2*Q] = V_next[1:2*Q+1] - V_next[:2*Q]
        
        # Valid asks: q > -Q => indices 1 to 2Q
        delta_a[1:] = V_next[:2*Q] - V_next[1:]
        
        # Solve for optimal depths (fast approximation)
        bid_depths, bid_vals = _solve_optimal_depths_vectorized(A, k, dt, delta_b)
        ask_depths, ask_vals = _solve_optimal_depths_vectorized(A, k, dt, delta_a)
        
        # Mask out invalid quotes
        bid_depths[2*Q] = np.inf
        bid_vals[2*Q] = 0.0
        
        ask_depths[0] = np.inf
        ask_vals[0] = 0.0
        
        # V_curr = -cost + V_next + contrib_b + contrib_a
        V_curr = -cost + V_next + bid_vals + ask_vals
        
        opt_bid[n, :] = bid_depths
        opt_ask[n, :] = ask_depths
        
        V_next = V_curr
    
    table = {'bid': opt_bid, 'ask': opt_ask}
    _OPT_TABLES[key] = table
    return table


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB solution.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple safe heuristic if running out of time
        sigma = params["sigma"]
        k = params["k"]
        Q = params["Q"]
        q = np.asarray(inventory, dtype=np.float64)
        base_depth = 1.0 / k
        bid_depth = np.full_like(q, base_depth)
        ask_depth = np.full_like(q, base_depth)
        bid_depth[q >= Q] = np.inf
        ask_depth[q <= -Q] = np.inf
        return bid_depth, ask_depth

    # Get or compute table
    table = _get_optimal_table(params)
    
    # Map t to step index
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    # Map inventory to index
    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.int64)
    # Clamp to [-Q, Q] just in case
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices]
    ask_depths = table['ask'][n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
