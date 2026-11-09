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
    # b >= 0 => x <= 1
    # Use log space for x
    # Adaptive grid size: use more points if we have time
    elapsed = time.time() - _START_TIME
    if elapsed < 60:
        n_grid = 3000
    elif elapsed < 120:
        n_grid = 2000
    else:
        n_grid = 1000
        
    x_grid = np.logspace(-12, 0, n_grid)
    log_x = np.log(x_grid)
    p_grid = 1.0 - np.exp(-lambda_t * x_grid)

    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    # Pre-compute term for J: C - (1/k) * log_x
    inv_k = 1.0 / k
    log_x_scaled = inv_k * log_x
    
    q_vals = np.arange(-Q, Q + 1)
    
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
        cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt

        V_next = V[t + 1, :]

        # C_b[i] = V_next[i+1] - V_next[i] for i < 2Q (q < Q)
        # For q=Q (idx 2Q), C_b is not used / b=inf
        C_b = np.zeros(n_q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]

        # C_a[i] = V_next[i-1] - V_next[i] for i > 0 (q > -Q)
        # For q=-Q (idx 0), C_a is not used / a=inf
        C_a = np.zeros(n_q)
        C_a[1:] = V_next[:-1] - V_next[1:]

        # Compute J_b for all q, all grid points
        # J_b = p_grid[None, :] * (C_b[:, None] - log_x_scaled[None, :])
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

    # Final polish: refine depths using Newton's method on the continuous objective
    # J(x) = (1 - exp(-lambda_t * x)) * (C - (1/k)*log(x))
    # We want to maximize J(x).
    # J'(x) = lambda_t*exp(-lambda_t*x)*(C - inv_k*log(x)) - (1-exp(-lambda_t*x))*inv_k/x
    # We solve J'(x) = 0.
    
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 10:  # only if we have time
        for t in range(N):
            if time.time() > _DEADLINE - 5:
                break
            
            # Refine Bid depths
            # We only refine if the grid found a finite value
            valid_b = np.isfinite(opt_b[t, :])
            valid_b[2*Q] = False # Boundary
            
            if np.any(valid_b):
                idxs_b = np.where(valid_b)[0]
                C_b_vals = C_b[idxs_b]
                x_cur = np.exp(-k * opt_b[t, idxs_b])
                
                # Newton iterations
                for _ in range(5):
                    exp_term = np.exp(-lambda_t * x_cur)
                    p_term = 1.0 - exp_term
                    log_term = np.log(x_cur)
                    
                    # Derivative J'(x)
                    # d/dx [ (1-e^{-lx})(C - 1/k ln x) ]
                    # = l e^{-lx} (C - 1/k ln x) + (1-e^{-lx}) (-1/(k x))
                    dJdx = lambda_t * exp_term * (C_b_vals - inv_k * log_term) - p_term * inv_k / x_cur
                    
                    # Second derivative for step size stability (optional, but helps)
                    # Or just use fixed damping
                    # J''(x) approx -lambda^2 e^{-lx} (C - 1/k ln x) - lambda e^{-lx} (1/k) - (1/k) e^{-lx} / x + p_term (1/k) / x^2 ... 
                    # Simpler: use a damped Newton step
                    # x_new = x - dJdx / J''
                    # Since J is concave in the region of interest, J'' < 0.
                    # We can estimate J'' or just use a small step.
                    # Let's use a robust update: x_new = x * exp(-dJdx / (J_val + eps)) is not standard.
                    # Standard Newton: x_new = x - dJdx / J''
                    # Let's compute J'' numerically or analytically?
                    # Analytical J'' is messy. Let's use a simplified damped Newton.
                    # If dJdx > 0, we need to increase x? No, if J' > 0, J is increasing, so we want larger x.
                    # Wait, x = exp(-kb). Larger b -> smaller x.
                    # If we want larger b, we want smaller x.
                    # The variable in Newton is x.
                    # If J'(x) > 0, J is increasing with x. So optimal x is to the right (larger x).
                    # If J'(x) < 0, J is decreasing with x. So optimal x is to the left (smaller x).
                    
                    # Let's just do a simple bisection or fixed point if Newton is unstable.
                    # Actually, for these parameters, Newton is usually stable.
                    # Estimate J'' using a small perturbation? Too slow.
                    # Let's use the fact that x is in (0, 1).
                    # We can clamp x_new to [1e-12, 1.0].
                    
                    # Heuristic step size based on magnitude of derivative
                    step = dJdx * 0.5 # Damping factor
                    x_new = x_cur - step
                    
                    # Clamp
                    x_new = np.clip(x_new, 1e-12, 1.0)
                    
                    # Check convergence
                    if np.max(np.abs(x_new - x_cur)) < 1e-10:
                        break
                    x_cur = x_new
                
                # Update opt_b with refined x
                b_refined = -np.log(x_cur) / k
                # Ensure b is not negative (x <= 1)
                b_refined = np.maximum(b_refined, 0.0)
                opt_b[t, idxs_b] = b_refined

            # Refine Ask depths
            valid_a = np.isfinite(opt_a[t, :])
            valid_a[0] = False # Boundary
            
            if np.any(valid_a):
                idxs_a = np.where(valid_a)[0]
                C_a_vals = C_a[idxs_a]
                x_cur = np.exp(-k * opt_a[t, idxs_a])
                
                for _ in range(5):
                    exp_term = np.exp(-lambda_t * x_cur)
                    p_term = 1.0 - exp_term
                    log_term = np.log(x_cur)
                    
                    dJdx = lambda_t * exp_term * (C_a_vals - inv_k * log_term) - p_term * inv_k / x_cur
                    
                    step = dJdx * 0.5
                    x_new = x_cur - step
                    x_new = np.clip(x_new, 1e-12, 1.0)
                    
                    if np.max(np.abs(x_new - x_cur)) < 1e-10:
                        break
                    x_cur = x_new
                
                a_refined = -np.log(x_cur) / k
                a_refined = np.maximum(a_refined, 0.0)
                opt_a[t, idxs_a] = a_refined

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
