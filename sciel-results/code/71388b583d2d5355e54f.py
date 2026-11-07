import numpy as np
import time

_policy_cache = {}
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _solve_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    risk_term = (gamma / 2.0) * sigma**2 * dt
    n_states = 2 * Q + 1
    
    V = np.zeros(n_states)
    bid_depth = np.full(n_states, 1.0 / k)
    ask_depth = np.full(n_states, 1.0 / k)
    
    d_min = -3.0 / k + 1e-6
    if A * dt > 1e-12:
        d_max = -np.log(1e-10 / (A * dt)) / k
        d_max = min(d_max, 5.0 / k)
    else:
        d_max = 5.0 / k
    
    n_grid = 100
    d_grid = np.linspace(d_min, d_max, n_grid)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
    
    max_iter = 500
    tol = 1e-10
    deadline = _get_start_time() + 320.0  # Safety margin
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        V_new = np.zeros(n_states)
        bid_depth_new = np.full(n_states, 1.0/k)
        ask_depth_new = np.full(n_states, 1.0/k)
        
        for i in range(n_states):
            q = i - Q
            
            if q == Q:
                bid_depth_new[i] = np.inf
                terms = p_grid * (V[i-1] - V[i] - d_grid)
                best_idx = np.argmax(terms)
                ask_depth_new[i] = d_grid[best_idx]
                V_new[i] = terms[best_idx] + V[i] - risk_term
            elif q == -Q:
                ask_depth_new[i] = np.inf
                terms = p_grid * (d_grid + V[i+1] - V[i])
                best_idx = np.argmax(terms)
                bid_depth_new[i] = d_grid[best_idx]
                V_new[i] = terms[best_idx] + V[i] - risk_term
            else:
                best_val = -np.inf
                best_ib = 0
                best_ia = 0
                
                for ib in range(n_grid):
                    pb = p_grid[ib]
                    terms_ia = p_grid * (V[i-1] - (1.0 - pb) * V[i] - d_grid)
                    ia = np.argmax(terms_ia)
                    val = pb * (d_grid[ib] + V[i+1]) + terms_ia[ia] + (1.0 - pb) * V[i] - risk_term
                    if val > best_val:
                        best_val = val
                        best_ib = ib
                        best_ia = ia
                
                bid_depth_new[i] = d_grid[best_ib]
                ask_depth_new[i] = d_grid[best_ia]
                V_new[i] = best_val
        
        diff = np.max(np.abs(V_new - V))
        V = V_new
        bid_depth = bid_depth_new
        ask_depth = ask_depth_new
        
        if diff < tol:
            break
    
    return bid_depth, ask_depth, V

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        bid_depth, ask_depth, V = _solve_policy(params)
        _policy_cache[key] = (bid_depth, ask_depth, V)
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    Q = params['Q']
    bid_depth, ask_depth, V = _get_policy(params)
    
    inv = np.asarray(inventory, dtype=np.int32)
    indices = np.clip(inv + Q, 0, 2 * Q)
    
    bid_out = bid_depth[indices].astype(np.float64)
    ask_out = ask_depth[indices].astype(np.float64)
    
    bid_out = np.where(np.isnan(bid_out), 1.0/params['k'], bid_out)
    ask_out = np.where(np.isnan(ask_out), 1.0/params['k'], ask_out)
    
    return bid_out, ask_out
