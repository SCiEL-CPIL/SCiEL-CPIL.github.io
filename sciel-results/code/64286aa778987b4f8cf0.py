import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 285.0
_cache = {}

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Grid for depth optimization
    # b = -1/k * ln(x), x in (0, 1]
    # p = 1 - exp(-A * x * dt)
    # We want to cover the range where p * (b + D) is significant.
    # For large b (small x), p is small.
    # For small b (large x), p is large but b is small.
    
    # Use a log-spaced grid for x to handle the wide range of optimal depths.
    # x from 1e-10 to 1.
    n_grid = 2000
    x_grid = np.logspace(-10, 0, n_grid)
    
    # Precompute p and b on the grid
    p_grid = 1.0 - np.exp(-A * dt * x_grid)
    b_grid = -np.log(x_grid) / k
    
    # Vectorized components
    # p_b shape (n_grid, 1)
    # b_b shape (n_grid, 1)
    # p_a shape (1, n_grid)
    # a_a shape (1, n_grid)
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    # Precompute 2D arrays for vectorized computation
    pb_2d = p_grid[:, None]  # (n_grid, 1)
    b_2d = b_grid[:, None]
    pa_2d = p_grid[None, :]  # (1, n_grid)
    a_2d = b_grid[None, :]
    
    # Note: b_grid and a_grid are the same since the model is symmetric for bid and ask depths
    # unless D_b and D_a differ.
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 10:
            # Fallback: use simple heuristic
            for tt in range(t, -1, -1):
                for i in range(n_q):
                    q = q_vals[i]
                    if q < Q:
                        opt_b[tt, i] = 1.0 / k
                    if q > -Q:
                        opt_a[tt, i] = 1.0 / k
            break
        
        V_next = V[t + 1, :]
        
        for i in range(n_q):
            q = q_vals[i]
            
            # Boundary conditions
            if i == n_q - 1:  # q = Q, no bid
                D_a = V_next[i - 1] - V_next[i]
                vals = p_grid * (b_grid + D_a)
                max_val = np.max(vals)
                if max_val <= 1e-12:
                    opt_a[t, i] = np.inf
                    contrib_a = 0.0
                else:
                    opt_a[t, i] = b_grid[np.argmax(vals)]
                    contrib_a = max_val
                
                cost = -0.5 * gamma * sigma**2 * q**2 * dt
                V[t, i] = V_next[i] + cost + contrib_a
                opt_b[t, i] = np.inf
                continue
                
            if i == 0:  # q = -Q, no ask
                D_b = V_next[i + 1] - V_next[i]
                vals = p_grid * (b_grid + D_b)
                max_val = np.max(vals)
                if max_val <= 1e-12:
                    opt_b[t, i] = np.inf
                    contrib_b = 0.0
                else:
                    opt_b[t, i] = b_grid[np.argmax(vals)]
                    contrib_b = max_val
                
                cost = -0.5 * gamma * sigma**2 * q**2 * dt
                V[t, i] = V_next[i] + cost + contrib_b
                opt_a[t, i] = np.inf
                continue
            
            # Interior: optimize both b and a
            D_b = V_next[i + 1] - V_next[i]
            D_a = V_next[i - 1] - V_next[i]
            D_cross = V_next[i + 1] - 2 * V_next[i] + V_next[i - 1]
            
            # Joint objective:
            # J = p_b (b + D_b) + p_a (a + D_a) + p_b p_a D_cross
            # Vectorized:
            J = pb_2d * (b_2d + D_b) + pa_2d * (a_2d + D_a) + pb_2d * pa_2d * D_cross
            
            max_idx = np.argmax(J)
            max_val = J.flat[max_idx]
            
            if max_val <= 1e-12:
                opt_b[t, i] = np.inf
                opt_a[t, i] = np.inf
                contrib = 0.0
            else:
                ib = max_idx // n_grid
                ia = max_idx % n_grid
                opt_b[t, i] = b_grid[ib]
                opt_a[t, i] = b_grid[ia]
                contrib = max_val
            
            cost = -0.5 * gamma * sigma**2 * q**2 * dt
            V[t, i] = V_next[i] + cost + contrib
    
    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        return np.full(4000, np.inf), np.full(4000, np.inf)
        
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs].copy()
    ask_depths = opt_a[t_idx, idxs].copy()
    
    # Ensure validity: no NaN, no -inf, no values < -3/k
    k = params['k']
    min_valid = -3.0 / k
    
    # Replace invalid values with inf
    for arr in [bid_depths, ask_depths]:
        arr[np.isnan(arr)] = np.inf
        arr[arr == -np.inf] = np.inf
        arr[arr < min_valid] = np.inf
    
    # Ensure non-negative (depths should be >= 0)
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
