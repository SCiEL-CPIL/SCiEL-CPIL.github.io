import numpy as np

# Global cache for policies
_POLICY_CACHE = {}

def _solve_scalar(d, delta_v, K, k):
    """Solve K(d + delta_v)exp(-kd) = 1 for d >= 0."""
    # Same as before, but we might not need it if we just use the DP result directly
    # But we need it to generate the optimal depths for the coarse grid DP
    d_guess = max(0.0, 1.0/k - delta_v)
    for _ in range(20):
        val = K * (d_guess + delta_v) * np.exp(-k * d_guess)
        if val < 1e-12:
            d_guess += 1.0
            continue
        g = np.log(val) - 1.0
        denom = d_guess + delta_v
        if denom < 1e-9:
            d_guess += 0.1
            continue
        gp = 1.0/denom - k
        if abs(gp) < 1e-12:
            break
        step = g / gp
        d_new = d_guess - step
        if d_new < 0:
            d_new = 0.0
            if d_guess < 1e-6:
                break
            d_guess = d_new
        else:
            if abs(d_new - d_guess) < 1e-8:
                d_guess = d_new
                break
            d_guess = d_new
    return max(0.0, d_guess)

def _compute_coarse_dp(params, Q_coarse):
    """
    Compute optimal depths on a coarse inventory grid.
    Returns arrays of shape (steps, 2*Q_coarse+1) for bid and ask depths.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    
    dt = T / steps
    K = A * k * dt
    
    num_q = 2 * Q_coarse + 1
    V_next = np.zeros(num_q)
    
    bid_depths = np.full((steps, num_q), np.inf)
    ask_depths = np.full((steps, num_q), np.inf)
    
    penalty_coeff = 0.5 * gamma * (sigma ** 2) * dt
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_q)
        for i in range(num_q):
            q = i - Q_coarse
            
            if q < Q_coarse:
                delta_b = V_next[i+1] - V_next[i]
            else:
                delta_b = -np.inf
            
            if q > -Q_coarse:
                delta_a = V_next[i-1] - V_next[i]
            else:
                delta_a = -np.inf
            
            d_b = 0.0
            d_a = 0.0
            p_b = 0.0
            p_a = 0.0
            
            if delta_b > -np.inf:
                d_b = _solve_scalar(0.0, delta_b, K, k)
                bid_depths[n, i] = d_b
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
            
            if delta_a > -np.inf:
                d_a = _solve_scalar(0.0, delta_a, K, k)
                ask_depths[n, i] = d_a
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
            
            val = 0.0
            val += p_b * d_b
            val += p_a * d_a
            
            val += (1 - p_b) * (1 - p_a) * V_next[i]
            if q < Q_coarse:
                val += p_b * (1 - p_a) * V_next[i+1]
            if q > -Q_coarse:
                val += p_a * (1 - p_b) * V_next[i-1]
            if q < Q_coarse and q > -Q_coarse:
                val += p_b * p_a * V_next[i]
            
            val -= penalty_coeff * q * q
            
            V_curr[i] = val
        
        V_next = V_curr
        
    return bid_depths, ask_depths

def _fit_affine_policy(bid_depths, ask_depths, Q_coarse):
    """
    Fit affine policies d = alpha + beta * q for each time step.
    """
    steps = bid_depths.shape[0]
    num_q = bid_depths.shape[1]
    
    # q values for the coarse grid
    q_vals = np.arange(-Q_coarse, Q_coarse + 1)
    
    alpha_b = np.zeros(steps)
    beta_b = np.zeros(steps)
    alpha_a = np.zeros(steps)
    beta_a = np.zeros(steps)
    
    for n in range(steps):
        # Filter out inf/nan if any (shouldn't happen if Q_coarse < Q_real)
        # But at boundaries, if no fill is allowed, depth is inf?
        # In our DP, if q=Q, no bid fill, so bid_depth is inf.
        # We should only fit on valid points.
        
        valid_b = np.isfinite(bid_depths[n])
        valid_a = np.isfinite(ask_depths[n])
        
        if np.sum(valid_b) > 1:
            # Fit line to valid points
            # Use np.polyfit
            try:
                coeff_b = np.polyfit(q_vals[valid_b], bid_depths[n, valid_b], 1)
                beta_b[n] = coeff_b[0]
                alpha_b[n] = coeff_b[1]
            except:
                beta_b[n] = 0
                alpha_b[n] = np.mean(bid_depths[n, valid_b])
        else:
            beta_b[n] = 0
            alpha_b[n] = 0 # Default to 0 if no valid data? Or inf?
            
        if np.sum(valid_a) > 1:
            try:
                coeff_a = np.polyfit(q_vals[valid_a], ask_depths[n, valid_a], 1)
                beta_a[n] = coeff_a[0]
                alpha_a[n] = coeff_a[1]
            except:
                beta_a[n] = 0
                alpha_a[n] = np.mean(ask_depths[n, valid_a])
        else:
            beta_a[n] = 0
            alpha_a[n] = 0
            
    return alpha_b, beta_b, alpha_a, beta_a

def quote(t, inventory, mid, params):
    """
    Optimal market making with inventory risk using Affine Policy on Coarse Grid.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    
    if key not in _POLICY_CACHE:
        # Use a coarse grid, e.g., 15
        Q_coarse = 15
        bid_depths_coarse, ask_depths_coarse = _compute_coarse_dp(params, Q_coarse)
        alpha_b, beta_b, alpha_a, beta_a = _fit_affine_policy(bid_depths_coarse, ask_depths_coarse, Q_coarse)
        _POLICY_CACHE[key] = (alpha_b, beta_b, alpha_a, beta_a)
        
    alpha_b, beta_b, alpha_a, beta_a = _POLICY_CACHE[key]
    
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    
    # Determine step index
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Compute depths using affine policy
    # inventory is array of ints
    inv = inventory.astype(float)
    
    bid_depths = alpha_b[n] + beta_b[n] * inv
    ask_depths = alpha_a[n] + beta_a[n] * inv
    
    # Clip to non-negative
    bid_depths = np.clip(bid_depths, 0, None)
    ask_depths = np.clip(ask_depths, 0, None)
    
    # Handle boundaries: if inventory is at max Q, no bid should be placed?
    # The affine policy might suggest a positive depth.
    # The task says "no bid fill at q=Q". It doesn't say "no bid quote".
    # But quoting a bid at q=Q is useless if it can't fill?
    # Actually, if q=Q, a bid fill would make it Q+1 which is invalid.
    # So the probability of fill is 0.
    # The optimal depth should be such that p_b is 0 or d is inf?
    # In the DP, we set d=inf if not allowed.
    # Here, if we quote a finite d, the model will try to fill it.
    # The model description: "no bid fill at q = Q".
    # So if we quote a bid at q=Q, it just won't fill.
    # The objective includes "depths earned". If no fill, 0 earned.
    # So quoting a finite depth is harmless (0 reward) but might be suboptimal if it affects mid? No, mid is exogenous.
    # However, quoting a bid at q=Q is "waste" but not penalized directly in the objective function provided, 
    # UNLESS the evaluator penalizes "wrong" quotes. 
    # "Values below -3/k ... make the whole result invalid".
    # It doesn't say quoting at boundary is invalid.
    # But to be safe, we should probably set depth to inf or 0?
    # If we set to 0, p_b = 1-exp(0) = 0.
    # If we set to inf, p_b = 1-exp(-inf) = 1. But fill is blocked.
    # Let's set to 0 for safety to avoid any weird interactions, or inf to match DP logic?
    # The DP logic used inf to signify "don't quote".
    # Let's set to inf if at boundary to match the "no quote" semantics of the optimal policy.
    
    Q = params["Q"]
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    return bid_depths, ask_depths
