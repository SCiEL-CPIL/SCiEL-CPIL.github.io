"""
Strategy: Pre-compute the optimal policy using Dynamic Programming (Bellman Equation)
with a higher resolution grid search to better approximate the continuous optimum.
Includes a time guard to ensure completion within the 330s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin

def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all (t, q) states using DP.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    V_next = np.zeros(n_states)
    
    # Higher resolution grid
    max_depth = 2.5
    n_grid = 250
    grid = np.linspace(0, max_depth, n_grid)
    
    # Precompute probabilities
    exp_term = np.exp(-k * grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Precompute 1-p
    one_minus_p = 1.0 - p_grid
    
    # Precompute b and a grids for broadcasting
    b_2d = grid[:, None]
    a_2d = grid[None, :]
    
    # We will iterate t from steps-1 down to 0
    for t in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _start_time > TIME_LIMIT:
            # Fallback: use current V_next to estimate actions? 
            # Or just return what we have (which might be incomplete).
            # Since we fill opt_bid/opt_ask as we go, if we break early, 
            # the remaining steps will have 0 depths.
            # This is better than crashing.
            break
            
        V_current = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_values[q_idx]
            
            allow_bid = (q < Q)
            allow_ask = (q > -Q)
            
            v_q = V_next[q_idx]
            v_qp = V_next[q_idx + 1] if allow_bid else 0.0
            v_qm = V_next[q_idx - 1] if allow_ask else 0.0
            
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Setup p arrays
            if allow_bid:
                pb = p_grid
                ompb = one_minus_p
            else:
                pb = np.zeros(n_grid)
                ompb = np.ones(n_grid)
                
            if allow_ask:
                pa = p_grid
                ompa = one_minus_p
            else:
                pa = np.zeros(n_grid)
                ompa = np.ones(n_grid)
            
            # Broadcasting
            pb_2d = pb[:, None]
            pa_2d = pa[None, :]
            ompb_2d = ompb[:, None]
            ompa_2d = ompa[None, :]
            
            # Terms
            # T1: No fills
            t1 = ompb_2d * ompa_2d * v_q
            
            # T2: Bid fill
            t2 = pb_2d * ompa_2d * (b_2d + v_qp)
            
            # T3: Ask fill
            t3 = ompb_2d * pa_2d * (a_2d + v_qm)
            
            # T4: Both fill
            t4 = pb_2d * pa_2d * (b_2d + a_2d + v_q)
            
            total = t1 + t2 + t3 + t4 - cost
            
            max_idx = np.argmax(total)
            i, j = np.unravel_index(max_idx, total.shape)
            
            opt_bid[t, q_idx] = grid[i]
            opt_ask[t, q_idx] = grid[j]
            V_current[q_idx] = total[i, j]
            
        V_next = V_current
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
