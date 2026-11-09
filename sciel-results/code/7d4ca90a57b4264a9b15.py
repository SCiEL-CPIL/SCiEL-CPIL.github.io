import numpy as np
import time

_cache = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 35s safety margin before 330s kill

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    # Deadline check: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        sigma = params['sigma']
        Q = params['Q']
        n_q = 2 * Q + 1
        N = params['steps']
        # Fallback: symmetric small spreads
        opt_b = np.full((N, n_q), 0.05)
        opt_a = np.full((N, n_q), 0.05)
        opt_b[:, 2 * Q] = np.inf   # no bid at max inventory
        opt_a[:, 0] = np.inf       # no ask at min inventory
        _cache[key] = (opt_b, opt_a)
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
    lambda_t = A * dt
    n_q = 2 * Q + 1

    # Grid for x = exp(-k*b)
    # Use a moderate grid size. The Newton polish will fix the precision.
    # 2000 points is usually sufficient for a good initial guess for Newton.
    n_grid = 2000
    x_grid = np.logspace(-12, 0, n_grid)
    log_x = np.log(x_grid)
    # p = 1 - exp(-lambda * x)
    p_grid = 1.0 - np.exp(-lambda_t * x_grid)

    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    # Pre-compute term for J: C - (1/k) * log_x
    inv_k = 1.0 / k
    log_x_scaled = inv_k * log_x
    
    # We need C_b and C_a for all t to do the vectorized polish later.
    # Storing them takes O(N*Q) memory, which is fine (1000*61 floats ~ 240KB).
    C_b_all = np.zeros((N, n_q))
    C_a_all = np.zeros((N, n_q))

    for t in range(N - 1, -1, -1):
        # Deadline check inside the loop
        if time.time() > _DEADLINE:
            # Fill remaining steps with simple fallback
            for tt in range(t, -1, -1):
                opt_b[tt, :] = 0.05
                opt_a[tt, :] = 0.05
                opt_b[tt, 2 * Q] = np.inf
                opt_a[tt, 0] = np.inf
            _cache[key] = (opt_b, opt_a)
            return _cache[key]

        # Compute cost and next values
        q_vals = np.arange(-Q, Q + 1)
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt

        V_next = V[t + 1, :]

        # C_b[i] = V_next[i+1] - V_next[i] for i < 2Q (q < Q)
        C_b = np.zeros(n_q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        C_b_all[t, :] = C_b

        # C_a[i] = V_next[i-1] - V_next[i] for i > 0 (q > -Q)
        C_a = np.zeros(n_q)
        C_a[1:] = V_next[:-1] - V_next[1:]
        C_a_all[t, :] = C_a

        # Compute J_b for all q, all grid points
        # Shape: (n_q, n_grid)
        J_b = p_grid[None, :] * (C_b[:, None] - log_x_scaled[None, :])

        # Find max
        max_idx_b = np.argmax(J_b, axis=1)
        max_val_b = np.max(J_b, axis=1)

        # Determine b values
        b_vals = -log_x_scaled[max_idx_b]

        # If max_val <= 0, it's better not to quote (b=inf)
        b_vals[max_val_b <= 1e-9] = np.inf

        # Boundary: q=Q (idx 2Q) must have b=inf
        b_vals[2*Q] = np.inf

        opt_b[t, :] = b_vals

        # Same for Ask
        J_a = p_grid[None, :] * (C_a[:, None] - log_x_scaled[None, :])

        max_idx_a = np.argmax(J_a, axis=1)
        max_val_a = np.max(J_a, axis=1)

        a_vals = -log_x_scaled[max_idx_a]
        a_vals[max_val_a <= 1e-9] = np.inf
        a_vals[0] = np.inf # q=-Q

        opt_a[t, :] = a_vals

        # Update V
        contrib_b = np.where(opt_b[t, :] == np.inf, 0.0, max_val_b)
        contrib_a = np.where(opt_a[t, :] == np.inf, 0.0, max_val_a)

        V[t, :] = V_next + cost + contrib_b + contrib_a

    # Final polish: Vectorized Newton refinement
    # We refine x for all (t, q) simultaneously.
    # x = exp(-k*b) => b = -log(x)/k
    # Objective to maximize: J(x) = (1 - exp(-lambda_t * x)) * (C - (1/k)*log(x))
    # dJ/dx = lambda_t * exp(-lambda_t * x) * (C - (1/k)*log(x)) - (1 - exp(-lambda_t * x)) * (1/(k*x))
    # d2J/dx2 = -lambda_t^2 * exp(-lambda_t * x) * (C - (1/k)*log(x)) + lambda_t * exp(-lambda_t * x) * (1/(k*x)) 
    #           - [ -lambda_t * exp(-lambda_t * x) * (1/(k*x)) + (1 - exp(-lambda_t * x)) * (-1/(k*x^2)) ]
    #           = -lambda_t^2 * exp(-lambda_t * x) * (C - (1/k)*log(x)) + lambda_t * exp(-lambda_t * x) * (1/(k*x)) 
    #           + lambda_t * exp(-lambda_t * x) * (1/(k*x)) + (1 - exp(-lambda_t * x)) * (1/(k*x^2))
    #           = -lambda_t^2 * exp(-lambda_t * x) * (C - (1/k)*log(x)) + 2 * lambda_t * exp(-lambda_t * x) * (1/(k*x)) 
    #           + (1 - exp(-lambda_t * x)) * (1/(k*x^2))

    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 20:
        # Prepare arrays for vectorized Newton
        # x_b shape: (N, n_q), x_a shape: (N, n_q)
        # We only refine where b/a is finite.
        
        x_b = np.exp(-k * opt_b)
        x_a = np.exp(-k * opt_a)
        
        # Mask for valid refinements
        valid_b = np.isfinite(opt_b)
        valid_a = np.isfinite(opt_a)
        
        # C_b_all and C_a_all are already computed
        # We need to handle the case where C is negative? 
        # The optimal x should always be in (0, 1].
        
        for _iter in range(30):
            if time.time() > _DEADLINE - 10:
                break
                
            # --- Refine Bid ---
            if np.any(valid_b):
                xb = x_b[valid_b]
                Cb = C_b_all[valid_b]
                
                exp_term = np.exp(-lambda_t * xb)
                p_term = 1.0 - exp_term
                log_xb = np.log(xb)
                
                # dJ/dx
                dJdx = lambda_t * exp_term * (Cb - inv_k * log_xb) - p_term * (inv_k / xb)
                
                # d2J/dx2
                d2Jdx2 = (-lambda_t**2 * exp_term * (Cb - inv_k * log_xb) 
                          + 2 * lambda_t * exp_term * (inv_k / xb) 
                          + p_term * (inv_k / (xb * xb)))
                
                # Newton step
                # Avoid division by zero or near-zero
                mask_nonzero = np.abs(d2Jdx2) > 1e-14
                xb_new = xb.copy()
                xb_new[mask_nonzero] = xb[mask_nonzero] - dJdx[mask_nonzero] / d2Jdx2[mask_nonzero]
                
                # Clamp
                xb_new = np.clip(xb_new, 1e-15, 1.0)
                
                x_b[valid_b] = xb_new
                
            # --- Refine Ask ---
            if np.any(valid_a):
                xa = x_a[valid_a]
                Ca = C_a_all[valid_a]
                
                exp_term = np.exp(-lambda_t * xa)
                p_term = 1.0 - exp_term
                log_xa = np.log(xa)
                
                # dJ/dx
                dJdx = lambda_t * exp_term * (Ca - inv_k * log_xa) - p_term * (inv_k / xa)
                
                # d2J/dx2
                d2Jdx2 = (-lambda_t**2 * exp_term * (Ca - inv_k * log_xa) 
                          + 2 * lambda_t * exp_term * (inv_k / xa) 
                          + p_term * (inv_k / (xa * xa)))
                          
                # Newton step
                mask_nonzero = np.abs(d2Jdx2) > 1e-14
                xa_new = xa.copy()
                xa_new[mask_nonzero] = xa[mask_nonzero] - dJdx[mask_nonzero] / d2Jdx2[mask_nonzero]
                
                # Clamp
                xa_new = np.clip(xa_new, 1e-15, 1.0)
                
                x_a[valid_a] = xa_new

        # Convert back to depths
        opt_b[valid_b] = -np.log(x_b[valid_b]) / k
        opt_a[valid_a] = -np.log(x_a[valid_a]) / k

    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
    if t_idx < 0:
        t_idx = 0
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs]
    ask_depths = opt_a[t_idx, idxs]
    
    return bid_depths, ask_depths
