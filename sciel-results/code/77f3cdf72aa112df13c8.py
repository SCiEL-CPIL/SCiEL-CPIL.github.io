"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _solve_depth_newton(A, k, dt, delta, d0, max_iter=50, tol=1e-14):
    """
    Solve the FOC for optimal depth d given inventory change delta.
    f(d) = p(d) * (d + delta) where p(d) = 1 - exp(-A*exp(-k*d)*dt).
    FOC: p'(d)*(d+delta) + p(d) = 0
    Uses Newton's method with careful handling of edge cases.
    """
    d = max(0.0, min(d0, 15.0 / k))
    
    for _ in range(max_iter):
        exp_kd = np.exp(-k * d)
        lam = A * exp_kd * dt
        
        if lam < 1e-10:
            # p ≈ lam, p' ≈ k*lam
            p = lam
            p_prime = k * lam
            p_double_prime = -k * k * lam
        else:
            exp_neg_lam = np.exp(-lam)
            p = 1.0 - exp_neg_lam
            p_prime = k * lam * exp_neg_lam
            p_double_prime = -k * k * lam * exp_neg_lam * (1.0 - lam)
        
        g = p_prime * (d + delta) + p
        g_prime = p_double_prime * (d + delta) + p_prime
        
        if abs(g_prime) < 1e-30:
            break
            
        newton_step = g / g_prime
        d_new = d - newton_step
        
        if d_new < 0:
            d_new = 0
        elif d_new > 15.0 / k:
            d_new = 15.0 / k
        
        if abs(d_new - d) < tol:
            d = d_new
            break
        d = d_new
    
    return d

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming with Newton-refined FOC solution."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
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
    
    # Value table: V[n, q] for n=0..steps, q=-Q..Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Grid for initial guesses
    d_max = 15.0 / k
    num_points = 400
    D_grid = np.linspace(0.0, d_max, num_points)
    
    # Precompute fill probabilities for the grid
    exp_term = np.exp(-k * D_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    def _optimal_depth(delta_val):
        """Find optimal depth for a given delta using grid search + Newton refinement."""
        if delta_val >= 0:
            # If delta is non-negative, the objective f(d) = p(d)*(d+delta) is likely increasing
            # or has a maximum at small d. Check boundary d=0 first.
            # Actually, let's just use the grid to find a good starting point.
            pass
        
        vals = p_grid * (D_grid + delta_val)
        max_val = np.max(vals)
        
        if max_val <= 0:
            return np.inf, 0.0
        
        idx = np.argmax(vals)
        d0 = D_grid[idx]
        
        # Newton refinement
        d_opt = _solve_depth_newton(A, k, dt, delta_val, d0)
        
        # Evaluate at the Newton solution
        exp_kd = np.exp(-k * d_opt)
        lam = A * exp_kd * dt
        if lam < 1e-10:
            p_opt = lam
        else:
            p_opt = 1.0 - np.exp(-lam)
        
        f_opt = p_opt * (d_opt + delta_val)
        
        # Safety: if Newton went wrong, fall back to grid best
        if f_opt < max_val - 1e-10:
            d_opt = d0
            f_opt = max_val
        
        return d_opt, max(f_opt, 0.0)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            # Fallback: use current V and no quotes for remaining steps
            break
        
        V_next = V[n + 1]
        
        # Calculate Delta V for bid and ask
        delta_b = np.zeros(2 * Q + 1)
        if Q > 0:
            delta_b[:-1] = V_next[1:] - V_next[:-1]
            # delta_b[-1] remains 0 (no bid at Q)
        
        delta_a = np.zeros(2 * Q + 1)
        if Q > 0:
            delta_a[1:] = V_next[:-1] - V_next[1:]
            # delta_a[0] remains 0 (no ask at -Q)
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        ask_depths = np.full(2 * Q + 1, np.inf)
        bid_rewards = np.zeros(2 * Q + 1)
        ask_rewards = np.zeros(2 * Q + 1)
        
        for q_idx in range(2 * Q + 1):
            # Bid side
            if q_idx < 2 * Q:  # Not at Q boundary
                d_b, r_b = _optimal_depth(delta_b[q_idx])
                bid_depths[q_idx] = d_b
                bid_rewards[q_idx] = r_b
            
            # Ask side
            if q_idx > 0:  # Not at -Q boundary
                d_a, r_a = _optimal_depth(delta_a[q_idx])
                ask_depths[q_idx] = d_a
                ask_rewards[q_idx] = r_a
        
        # Handle boundary conditions explicitly
        if Q > 0:
            bid_depths[-1] = np.inf
            bid_rewards[-1] = 0.0
            ask_depths[0] = np.inf
            ask_rewards[0] = 0.0
        
        # Update Value
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    # Deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        n_paths = len(inventory)
        return np.full(n_paths, np.inf), np.full(n_paths, np.inf)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to indices
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
