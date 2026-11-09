import numpy as np
import time

_policy_cache = {}
_deadline = 300

def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    n_levels = 2 * Q + 1
    V = np.zeros((n_levels, steps + 1))
    opt_bid = np.zeros((n_levels, steps))
    opt_ask = np.zeros((n_levels, steps))
    
    max_depth = 5.0 / k
    n_grid = 60
    depth_grid = np.linspace(0, max_depth, n_grid)
    B_grid, A_grid = np.meshgrid(depth_grid, depth_grid, indexing='ij')
    B_flat = B_grid.flatten()
    A_flat = A_grid.flatten()
    
    p_b_grid = 1 - np.exp(-A * np.exp(-k * B_flat) * dt)
    p_a_grid = 1 - np.exp(-A * np.exp(-k * A_flat) * dt)
    
    t0 = time.time()
    for n in range(steps - 1, -1, -1):
        if time.time() - t0 > _deadline - 60:
            for q_idx in range(n_levels):
                opt_bid[q_idx, n] = 1.0 / k
                opt_ask[q_idx, n] = 1.0 / k
            break
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            V_q = V[q_idx, n + 1]
            V_qm1 = V[q_idx - 1, n + 1] if q > -Q else 0.0
            V_qp1 = V[q_idx + 1, n + 1] if q < Q else 0.0
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            p_b = p_b_grid.copy()
            p_a = p_a_grid.copy()
            
            if q == Q:
                p_b[:] = 0.0
            if q == -Q:
                p_a[:] = 0.0
            
            vals = (p_b * B_flat + p_a * A_flat + 
                    p_b * (1.0 - p_a) * V_qm1 + 
                    (1.0 - p_b) * p_a * V_qp1 + 
                    ((1.0 - p_b) * (1.0 - p_a) + p_b * p_a) * V_q - cost)
            
            best_idx = np.argmax(vals)
            opt_bid[q_idx, n] = B_flat[best_idx]
            opt_ask[q_idx, n] = A_flat[best_idx]
            V[q_idx, n] = vals[best_idx]
    
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        _policy_cache[cache_key] = _precompute_policy(sigma, A, k, gamma, T, steps, Q)
    
    opt_bid, opt_ask = _policy_cache[cache_key]
    
    inv = np.asarray(inventory, dtype=np.int64)
    q_idx = inv + Q
    bid_depths = opt_bid[q_idx, n].copy()
    ask_depths = opt_ask[q_idx, n].copy()
    
    bid_depths[inv == Q] = np.inf
    ask_depths[inv == -Q] = np.inf
    
    return bid_depths, ask_depths
