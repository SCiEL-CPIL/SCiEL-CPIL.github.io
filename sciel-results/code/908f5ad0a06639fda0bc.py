# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 310.0  # Safety margin before 330s kill


def _newton_depth(c, k, M, d_init, max_iter=30):
    """
    Newton-Raphson to solve f'(d) = 0 for maximizing f(d) = p(d)*(d+M)
    where p(d) = 1 - exp(-c * exp(-k*d))
    
    f'(d) = c*k*exp(-2*k*d)*(d+M) + 1 - exp(-c*exp(-k*d))
    f''(d) = c*k*exp(-2*k*d)*(-2*k*(d+M) + 1) + c*k*exp(-k*d)*exp(-c*exp(-k*d))
    """
    d = d_init
    for _ in range(max_iter):
        e1 = np.exp(-k * d)
        e2 = e1 * e1  # exp(-2*k*d)
        p = 1.0 - np.exp(-c * e1)
        
        # f'(d)
        fp = c * k * e2 * (d + M) + p
        
        # f''(d)
        fpp = c * k * e2 * (1.0 - 2.0 * k * (d + M)) + c * k * e1 * np.exp(-c * e1)
        
        if abs(fpp) < 1e-15:
            break
        
        delta = fp / fpp
        d_new = d - delta
        
        # Keep in valid range
        if d_new < 1e-6:
            d_new = 1e-6
        if d_new > 50.0 / k:
            d_new = 50.0 / k
            
        if abs(d_new - d) < 1e-10:
            d = d_new
            break
        d = d_new
    
    return max(d, 1e-6)


def _solve_mdp_polynomial(params):
    """
    Solves the MDP using a hybrid approach:
    1. Coarse backward DP to get initial value estimates
    2. Forward Gauss-Seidel with polynomial value approximation
    3. Newton-Raphson for continuous depth optimization
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    c = A * dt  # Constant in fill probability
    n_inv = 2 * Q + 1
    
    # Phase 1: Coarse backward DP for initialization
    # Use a moderate grid for initial policy
    d_grid = np.linspace(0.01, 12.0 / k, 200)
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-c * exp_term)
    
    V_coarse = np.zeros((steps + 1, n_inv))
    D_b_coarse = np.full((steps, n_inv), 1.0 / k)
    D_a_coarse = np.full((steps, n_inv), 1.0 / k)
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 20:
            break
        for q_idx in range(n_inv):
            q = q_idx - Q
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            if q < Q:
                M_b = V_coarse[n + 1, q_idx + 1]
                obj_b = P_grid * (d_grid + M_b)
                idx_b = np.argmax(obj_b)
                D_b_coarse[n, q_idx] = d_grid[idx_b]
                val_b = obj_b[idx_b]
            else:
                val_b = 0.0
                
            if q > -Q:
                M_a = V_coarse[n + 1, q_idx - 1]
                obj_a = P_grid * (d_grid + M_a)
                idx_a = np.argmax(obj_a)
                D_a_coarse[n, q_idx] = d_grid[idx_a]
                val_a = obj_a[idx_a]
            else:
                val_a = 0.0
                
            V_coarse[n, q_idx] = val_b + val_a - cost
    
    # Phase 2: Forward Gauss-Seidel with Newton-Raphson refinement
    # Use the coarse DP as initialization, then refine with continuous optimization
    V = V_coarse.copy()
    D_b = D_b_coarse.copy()
    D_a = D_a_coarse.copy()
    
    # Iterate forward a few times to refine
    n_refine_passes = 3
    for pass_idx in range(n_refine_passes):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        
        for n in range(steps):
            if time.time() - _START_TIME > TIME_LIMIT - 5:
                break
                
            for q_idx in range(n_inv):
                q = q_idx - Q
                cost = 0.5 * gamma * sigma * sigma * q * q * dt
                
                # Bid side
                if q < Q:
                    M_b = V[n + 1, q_idx + 1]
                    # Use current depth as initial guess for Newton
                    d_init = D_b[n, q_idx]
                    d_b_opt = _newton_depth(c, k, M_b, d_init)
                    p_opt = 1.0 - np.exp(-c * np.exp(-k * d_b_opt))
                    val_b = p_opt * (d_b_opt + M_b)
                    D_b[n, q_idx] = d_b_opt
                else:
                    val_b = 0.0
                    D_b[n, q_idx] = np.inf
                    
                # Ask side
                if q > -Q:
                    M_a = V[n + 1, q_idx - 1]
                    d_init = D_a[n, q_idx]
                    d_a_opt = _newton_depth(c, k, M_a, d_init)
                    p_opt = 1.0 - np.exp(-c * np.exp(-k * d_a_opt))
                    val_a = p_opt * (d_a_opt + M_a)
                    D_a[n, q_idx] = d_a_opt
                else:
                    val_a = 0.0
                    D_a[n, q_idx] = np.inf
                
                V[n, q_idx] = val_b + val_a - cost
    
    # Phase 3: Final backward pass to ensure consistency
    # Re-compute V from the refined depths going backward
    V_final = np.zeros((steps + 1, n_inv))
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 2:
            break
        for q_idx in range(n_inv):
            q = q_idx - Q
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            if q < Q:
                M_b = V_final[n + 1, q_idx + 1]
                d_b = D_b[n, q_idx]
                p_b = 1.0 - np.exp(-c * np.exp(-k * d_b))
                val_b = p_b * (d_b + M_b)
            else:
                val_b = 0.0
                
            if q > -Q:
                M_a = V_final[n + 1, q_idx - 1]
                d_a = D_a[n, q_idx]
                p_a = 1.0 - np.exp(-c * np.exp(-k * d_a))
                val_a = p_a * (d_a + M_a)
            else:
                val_a = 0.0
                
            V_final[n, q_idx] = val_b + val_a - cost
    
    return D_b, D_a


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a key for the cache
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Check if we have already solved this MDP
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_mdp_polynomial(params)
    
    D_b, D_a = _POLICY_CACHE[key]
    
    # Determine current step n
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    # Process inventory
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Map inventory to index: q + Q
    q_indices = inv + Q
    # Clamp to valid range
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    # Apply boundary constraints explicitly
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    # Ensure depths are valid: > -3/k, not NaN
    min_depth = -3.0 / k + 1e-9
    bid_depths[bid_depths <= min_depth] = np.inf
    ask_depths[ask_depths <= min_depth] = np.inf
    bid_depths[np.isnan(bid_depths)] = np.inf
    ask_depths[np.isnan(ask_depths)] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
