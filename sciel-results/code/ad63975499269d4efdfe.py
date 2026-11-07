# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for learned affine policies per (sigma, A, k, gamma, Q, steps)
_policy_cache = {}
_start_time = time.time()


def _solve_depth_for_cost(C, k, A, dt):
    """
    Solve for optimal depth d given marginal cost C.
    Maximizes: d * p(d) - C * p(d) = p(d) * (d - C)
    where p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    FOC: exp(L) - 1 = k*L*(d - C), L = A*exp(-k*d)*dt
    """
    C = float(C)
    if C <= 0:
        return 0.0
    
    # Newton's method
    d = C + 1.0 / k
    d = np.clip(d, 0.0, 30.0 / k)
    
    for _ in range(20):
        L = A * np.exp(-k * d) * dt
        g = np.exp(L) - 1.0 - k * L * (d - C)
        gp = k * L * (k * (d - C) - np.exp(L) - 1.0)
        
        if abs(gp) < 1e-15:
            break
        
        d_new = d - g / gp
        d_new = np.clip(d_new, 0.0, 30.0 / k)
        
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    
    return float(d)


def _learn_affine_policy(sigma, A, k, gamma, Q, steps, T=1.0, n_iter=200):
    """
    Learn a time-homogeneous affine policy d(q) = alpha + beta * q
    via value iteration on the inventory state space.
    
    The policy is symmetric: bid_depth(q) = max(0, alpha + beta*q), ask_depth(q) = max(0, alpha - beta*q)
    where beta >= 0 means we quote wider when inventory is away from zero.
    
    Returns: (alpha, beta)
    """
    dt = T / steps
    nq = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Initialize value function
    V = np.zeros(nq, dtype=np.float64)
    
    # Initial policy guess
    alpha = 1.0 / k
    beta = 0.0
    
    for iteration in range(n_iter):
        # Compute depths from current policy
        bid_d = np.maximum(0.0, alpha + beta * q_vals)
        ask_d = np.maximum(0.0, alpha - beta * q_vals)
        
        # Boundary conditions
        bid_d[Q + Q] = np.inf  # q = Q
        ask_d[0 - Q + Q] = np.inf  # q = -Q
        
        # Compute probabilities
        L_b = np.where(np.isfinite(bid_d), A * np.exp(-k * bid_d) * dt, 0.0)
        L_a = np.where(np.isfinite(ask_d), A * np.exp(-k * ask_d) * dt, 0.0)
        
        p_b = 1.0 - np.exp(-L_b)
        p_a = 1.0 - np.exp(-L_a)
        
        # Rewards
        reward_b = np.where(np.isfinite(bid_d), bid_d * p_b, 0.0)
        reward_a = np.where(np.isfinite(ask_d), ask_d * p_a, 0.0)
        
        # Expected next value
        V_q = V.copy()
        V_q_plus = np.zeros(nq)
        V_q_minus = np.zeros(nq)
        
        V_q_plus[:nq-1] = V[1:]
        V_q_plus[nq-1] = V[nq-1]
        
        V_q_minus[1:] = V[:nq-1]
        V_q_minus[0] = V[0]
        
        E_V = (1.0 - p_b) * (1.0 - p_a) * V_q + \
               p_b * (1.0 - p_a) * V_q_plus + \
               (1.0 - p_b) * p_a * V_q_minus + \
               p_b * p_a * V_q
        
        # Penalty
        penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_new = -penalty + reward_b + reward_a + E_V
        
        # Update value function (with relaxation for stability)
        V = 0.5 * V + 0.5 * V_new
        
        # Update policy: for each q, find optimal depth given current V
        # Then fit affine policy to these optimal depths
        C_b_arr = np.zeros(nq)
        C_a_arr = np.zeros(nq)
        
        C_b_arr[:nq-1] = V_new[:nq-1] - V_new[1:]
        C_a_arr[1:] = V_new[1:] - V_new[:nq-1]
        
        # Solve for optimal depths at each q
        opt_bid = np.zeros(nq)
        opt_ask = np.zeros(nq)
        
        for i in range(nq):
            if i < nq - 1:  # q < Q
                opt_bid[i] = _solve_depth_for_cost(C_b_arr[i], k, A, dt)
            else:
                opt_bid[i] = np.inf
            
            if i > 0:  # q > -Q
                opt_ask[i] = _solve_depth_for_cost(C_a_arr[i], k, A, dt)
            else:
                opt_ask[i] = np.inf
        
        # Fit affine policy to optimal depths
        # For bid: d_b(q) ≈ alpha + beta*q for q in [-Q, Q-1]
        # For ask: d_a(q) ≈ alpha - beta*q for q in [-Q+1, Q]
        
        # Use finite values for fitting
        mask_bid = np.isfinite(opt_bid)
        mask_ask = np.isfinite(opt_ask)
        
        if np.sum(mask_bid) >= 2:
            q_fit = q_vals[mask_bid]
            d_fit = opt_bid[mask_bid]
            # Linear regression: d = alpha + beta * q
            A_mat = np.vstack([np.ones_like(q_fit), q_fit]).T
            coeffs_b = np.linalg.lstsq(A_mat, d_fit, rcond=None)[0]
            alpha_b = coeffs_b[0]
            beta_b = coeffs_b[1]
        else:
            alpha_b = alpha
            beta_b = beta
        
        if np.sum(mask_ask) >= 2:
            q_fit = q_vals[mask_ask]
            d_fit = opt_ask[mask_ask]
            # d = alpha - beta * q => d = alpha + (-beta) * q
            A_mat = np.vstack([np.ones_like(q_fit), q_fit]).T
            coeffs_a = np.linalg.lstsq(A_mat, d_fit, rcond=None)[0]
            alpha_a = coeffs_a[0]
            beta_a = -coeffs_a[1]
        else:
            alpha_a = alpha
            beta_a = beta
        
        # Average the estimates
        alpha_new = 0.5 * alpha_b + 0.5 * alpha_a
        beta_new = 0.5 * beta_b + 0.5 * beta_a
        
        # Ensure beta >= 0 (wider quotes when inventory is away from zero)
        beta_new = max(0.0, beta_new)
        alpha_new = max(0.0, alpha_new)
        
        # Check convergence
        if abs(alpha_new - alpha) < 1e-6 and abs(beta_new - beta) < 1e-6:
            alpha = alpha_new
            beta = beta_new
            break
        
        alpha = alpha_new
        beta = beta_new
    
    return alpha, beta


def quote(t, inventory, mid, params):
    """Quote using learned affine policy d(q) = alpha + beta*q."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    steps = params["steps"]
    T = params["T"]
    
    # Deadline guard
    if time.time() - _start_time > 295:
        # Return conservative default depths
        default_depth = 1.0 / k
        bid_depths = np.full(4000, default_depth, dtype=np.float64)
        ask_depths = np.full(4000, default_depth, dtype=np.float64)
        bid_depths[inventory >= Q] = np.inf
        ask_depths[inventory <= -Q] = np.inf
        return bid_depths, ask_depths
    
    # Check cache
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _policy_cache:
        _policy_cache[key] = _learn_affine_policy(sigma, A, k, gamma, Q, steps, T)
    
    alpha, beta = _policy_cache[key]
    
    q = inventory.astype(np.float64)
    
    # Compute depths: bid = alpha + beta*q, ask = alpha - beta*q
    bid_depths = alpha + beta * q
    ask_depths = alpha - beta * q
    
    # Clamp to valid range [0, 30/k]
    max_depth = 30.0 / k
    bid_depths = np.clip(bid_depths, 0.0, max_depth)
    ask_depths = np.clip(ask_depths, 0.0, max_depth)
    
    # Boundary conditions: no bid at q=Q, no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure validity
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
