import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0
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
    
    # Exact DP with coupled bid/ask optimization
    # State: inventory q in [-Q, Q]
    # Action: bid depth b >= 0, ask depth a >= 0
    # p_b = 1 - exp(-A * exp(-k*b) * dt), p_a = 1 - exp(-A * exp(-k*a) * dt)
    # V[t,q] = max_{b,a} { p_b*b + p_a*a - (gamma/2)*sigma^2*q^2*dt 
    #           + p_b*V[t+1,q+1] + p_a*V[t+1,q-1] + (1-p_b)*(1-p_a)*V[t+1,q] }
    # Boundary: q=Q -> no bid (p_b=0), q=-Q -> no ask (p_a=0)
    
    # Use a grid for depths. Since b and a are coupled, we need joint optimization.
    # But the objective is separable in a clever way. Let's use a 2D grid for (b, a).
    # However, 2D grid is expensive. Let's try a different approach.
    # 
    # Actually, we can optimize b and a jointly using a 2D grid.
    # Grid size: 100x100 = 10000 points per (t, q).
    # 1000 steps * 61 states * 10000 points = 610 million operations. Too slow.
    #
    # Alternative: use a smaller grid, e.g., 50x50 = 2500 points.
    # 1000 * 61 * 2500 = 152.5 million. Still might be slow.
    #
    # Let's try a smarter approach: for each (t, q), optimize b and a using a 2D grid 
    # but with a coarse resolution, then refine with local search.
    #
    # Or, we can use the fact that the objective is concave in b and a (approximately) 
    # and use gradient-based optimization.
    #
    # For now, let's use a 2D grid with 100x100 points.
    # We'll vectorize over q and t as much as possible.
    
    # Grid for depths: from 0 to some max value
    # b_max: when p_b is very small, b is large. Let's cap at 10/k or so.
    # For k=1.5, b_max ~ 6.7. For k=0.75, b_max ~ 13.3.
    # Let's use b_max = 15/k to be safe.
    
    b_max = 15.0 / k
    a_max = 15.0 / k
    
    # Number of grid points
    n_grid = 80  # 80x80 = 6400 points
    
    b_grid = np.linspace(0, b_max, n_grid)
    a_grid = np.linspace(0, a_max, n_grid)
    
    # Precompute p_b and p_a on the grid
    p_b_grid = 1.0 - np.exp(-A * np.exp(-k * b_grid) * dt)  # shape (n_grid,)
    p_a_grid = 1.0 - np.exp(-A * np.exp(-k * a_grid) * dt)  # shape (n_grid,)
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
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
        
        # V_next = V[t+1, :] shape (n_q,)
        V_next = V[t + 1, :]
        
        # For each q, compute the optimal (b, a)
        # We'll do this in a loop over q for clarity, but vectorize where possible
        
        for i in range(n_q):
            q = q_vals[i]
            
            # Boundary conditions
            if i == n_q - 1:  # q = Q, no bid
                # Optimize a only
                # V[t, Q] = max_a { p_a*a + (1-p_a)*V[t+1, Q-1] + p_a*V[t+1, Q] - cost }
                # Wait, if q=Q, bid fill is not allowed, so p_b=0.
                # V[t, Q] = max_a { p_a*a - (gamma/2)*sigma^2*Q^2*dt 
                #           + p_a*V[t+1, Q-1] + (1-p_a)*V[t+1, Q] }
                # = -cost + max_a { p_a*(a + V[t+1, Q-1] - V[t+1, Q]) + V[t+1, Q] }
                # = -cost + V[t+1, Q] + max_a p_a*(a + V[t+1, Q-1] - V[t+1, Q])
                
                D_a = V_next[i - 1] - V_next[i]  # V[t+1, Q-1] - V[t+1, Q]
                # Optimize a: max_a p_a(a) * (a + D_a)
                vals = p_a_grid * (a_grid + D_a)
                max_val = np.max(vals)
                if max_val <= 1e-12:
                    opt_a[t, i] = np.inf
                    contrib_a = 0.0
                else:
                    opt_a[t, i] = a_grid[np.argmax(vals)]
                    contrib_a = max_val
                
                cost = -0.5 * gamma * sigma**2 * q**2 * dt
                V[t, i] = V_next[i] + cost + contrib_a
                opt_b[t, i] = np.inf
                continue
                
            if i == 0:  # q = -Q, no ask
                # Optimize b only
                D_b = V_next[i + 1] - V_next[i]  # V[t+1, -Q+1] - V[t+1, -Q]
                vals = p_b_grid * (b_grid + D_b)
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
            D_b = V_next[i + 1] - V_next[i]  # V[t+1, q+1] - V[t+1, q]
            D_a = V_next[i - 1] - V_next[i]  # V[t+1, q-1] - V[t+1, q]
            D_cross = V_next[i + 1] - 2 * V_next[i] + V_next[i - 1]
            
            # Joint objective:
            # J(b,a) = p_b*b + p_a*a + p_b*D_b + p_a*D_a + p_b*p_a*D_cross
            #        = p_b*(b + D_b) + p_a*(a + D_a) + p_b*p_a*D_cross
            
            # 2D grid search
            pb_2d = p_b_grid[:, None]  # (n_grid, 1)
            pa_2d = p_a_grid[None, :]  # (1, n_grid)
            b_2d = b_grid[:, None]
            a_2d = a_grid[None, :]
            
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
                opt_a[t, i] = a_grid[ia]
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
