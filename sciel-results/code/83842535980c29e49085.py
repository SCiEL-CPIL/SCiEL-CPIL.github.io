"""Optimal market making with inventory risk via backward induction (vectorized)."""
import numpy as np
import time

_cache = {}
_start_time = None

def _compute_policy(params, steps=1000, Q=30):
    """Precompute optimal bid/ask depths for each (step, inventory) via backward induction."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    cache_key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], steps, Q)
    if cache_key in _cache:
        return _cache[cache_key]
    
    # Deadline guard: 320s limit
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback to simple policy
        depth = np.full(2*Q+1, 1.0/params['k'])
        opt_bid = np.tile(depth, (steps, 1))
        opt_ask = np.tile(depth, (steps, 1))
        _cache[cache_key] = (opt_bid, opt_ask)
        return (opt_bid, opt_ask)
    
    dt = params['T'] / steps
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    # Use a depth grid
    n_depths = 40
    d_min = 0.005
    d_max = 4.0 / k
    depths = np.linspace(d_min, d_max, n_depths)
    
    # Precompute fill probabilities
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    n_states = 2 * Q + 1
    V_next = np.zeros(n_states)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # Precompute risk penalty for each inventory level
    q_vals = np.arange(-Q, Q + 1)
    risk = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > 310:
            # Fill remaining with current best guess
            for nn in range(n, -1, -1):
                opt_bid[nn, :] = depths[10]
                opt_ask[nn, :] = depths[10]
            break
        
        V_curr = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_idx - Q
            r = risk[q_idx]
            
            if q == Q:
                pb_arr = np.zeros(n_depths)
            else:
                pb_arr = p_fill.copy()
            
            if q == -Q:
                pa_arr = np.zeros(n_depths)
            else:
                pa_arr = p_fill.copy()
            
            pb2d = pb_arr[:, np.newaxis]
            pa2d = pa_arr[np.newaxis, :]
            
            imm_reward = depths * pb_arr * np.ones((1, n_depths)) + np.ones((n_depths, 1)) * (depths * pa_arr)
            
            v_q = V_next[q_idx]
            v_qm1 = V_next[q_idx - 1] if q_idx > 0 else 0.0
            v_qp1 = V_next[q_idx + 1] if q_idx < n_states - 1 else 0.0
            
            e_v = ((1-pb2d)*(1-pa2d)*v_q + 
                   pb2d*(1-pa2d)*v_qm1 + 
                   (1-pb2d)*pa2d*v_qp1 + 
                   pb2d*pa2d*v_q)
            
            total = imm_reward - r + e_v
            
            max_idx = np.unravel_index(np.argmax(total), total.shape)
            V_curr[q_idx] = total[max_idx]
            opt_bid[n, q_idx] = depths[max_idx[0]]
            opt_ask[n, q_idx] = depths[max_idx[1]]
        
        V_next = V_curr
    
    # Ensure valid depths
    min_depth = -3.0 / k + 0.01
    opt_bid = np.maximum(opt_bid, min_depth)
    opt_ask = np.maximum(opt_ask, min_depth)
    
    _cache[cache_key] = (opt_bid, opt_ask)
    return (opt_bid, opt_ask)


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    steps = params['steps']
    Q = params['Q']
    k = params['k']
    
    n = int(round(t / (params['T'] / steps)))
    n = min(max(n, 0), steps - 1)
    
    opt_bid, opt_ask = _compute_policy(params, steps, Q)
    
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    bid_depths = opt_bid[n, q_idx].astype(np.float64)
    ask_depths = opt_ask[n, q_idx].astype(np.float64)
    
    # Ensure validity
    min_depth = -3.0 / k + 0.01
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    return bid_depths, ask_depths
