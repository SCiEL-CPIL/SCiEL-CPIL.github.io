import numpy as np
from functools import lru_cache

# Cache for precomputed policies
_policy_cache = {}

def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for each (t, q).
    Returns two arrays of shape (steps, 2*Q+1) for bid and ask depths.
    """
    dt = T / steps
    
    # Use a finer grid for better accuracy.
    # Depth range: [0.01, 5.0/k] is a safe range.
    # 50 points should be sufficient for high accuracy.
    n_depths = 50
    min_depth = 0.01
    max_depth = 5.0 / k
    depths = np.linspace(min_depth, max_depth, n_depths)
    
    # Precompute probabilities for all depths
    # p = 1 - exp(-A * exp(-k * d) * dt)
    p_depths = 1 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Precompute meshgrids for bid and ask depths
    # B[i, j] = depths[i], A_depth[i, j] = depths[j]
    B, A_depth = np.meshgrid(depths, depths, indexing='ij')
    PB, PA = np.meshgrid(p_depths, p_depths, indexing='ij')
    
    # Precompute immediate rewards
    rew_b = PB * B
    rew_a = PA * A_depth
    
    # Precompute transition probabilities
    prob_q_plus = (1 - PA) * PB
    prob_q_minus = (1 - PB) * PA
    prob_q = PB * PA + (1 - PB) * (1 - PA)
    
    # V[t, q] where q index is q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    # Iterate backwards from t = steps - 1 down to 0
    for t in range(steps - 1, -1, -1):
        for qi in range(-Q, Q + 1):
            q = qi
            q_idx = q + Q
            
            # Penalty for current inventory
            penalty = -0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get V(t+1, ...)
            V_next_q = V[t + 1, q_idx]
            
            # Handle boundaries for V_next
            if q < Q:
                V_next_q_plus = V[t + 1, q_idx + 1]
            else:
                V_next_q_plus = 0.0
                
            if q > -Q:
                V_next_q_minus = V[t + 1, q_idx - 1]
            else:
                V_next_q_minus = 0.0
            
            # Compute V_next and Total Value
            if q == Q:
                # No bid fill allowed. Effectively PB = 0.
                # prob_q_plus = 0
                # prob_q_minus = PA (since 1-PB=1)
                # prob_q = 1 - PA
                # rew_b = 0
                # rew_a = rew_a (unchanged)
                # V_next = PA * V_next_q_minus + (1 - PA) * V_next_q
                V_next = PA * V_next_q_minus + (1 - PA) * V_next_q
                total_val = penalty + rew_a + V_next
                # Note: rew_b is ignored because PB=0 implies no bid reward.
                # Actually, rew_b = PB * B. If PB=0, rew_b=0.
                # So we can just use total_val = penalty + rew_a + V_next.
                
            elif q == -Q:
                # No ask fill allowed. Effectively PA = 0.
                # prob_q_minus = 0
                # prob_q_plus = PB (since 1-PA=1)
                # prob_q = 1 - PB
                # rew_a = 0
                # rew_b = rew_b (unchanged)
                # V_next = PB * V_next_q_plus + (1 - PB) * V_next_q
                V_next = PB * V_next_q_plus + (1 - PB) * V_next_q
                total_val = penalty + rew_b + V_next
                
            else:
                # Interior states
                V_next = prob_q_plus * V_next_q_plus + prob_q_minus * V_next_q_minus + prob_q * V_next_q
                total_val = penalty + rew_b + rew_a + V_next
            
            # Find max
            max_val = np.max(total_val)
            idx = np.argmax(total_val)
            i, j = np.unravel_index(idx, total_val.shape)
            
            best_b = depths[i]
            best_a = depths[j]
            
            V[t, q_idx] = max_val
            opt_bid[t, q_idx] = best_b
            opt_ask[t, q_idx] = best_a
            
            # Handle boundary quotes
            if q == Q:
                opt_bid[t, q_idx] = np.inf
            if q == -Q:
                opt_ask[t, q_idx] = np.inf
                
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        # Precompute policy
        opt_bid, opt_ask = _precompute_policy(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (opt_bid, opt_ask)
    else:
        opt_bid, opt_ask = _policy_cache[key]
        
    # Get current step index
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Get inventory indices
    # inventory is array of ints
    q_idx = inventory + Q
    
    # Fetch depths
    d_bid = opt_bid[n, q_idx]
    d_ask = opt_ask[n, q_idx]
    
    # Ensure shapes and types
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    return d_bid, d_ask
