"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _solve_depth_foc(delta, A_dt, k, depth_min, depth_max):
    """
    Solve the first-order condition for optimal depth d:
    p(d) = (d + delta) * k * A_dt * exp(-k*d)
    where p(d) = 1 - exp(-A_dt * exp(-k*d))
    
    Using Newton's method on f(d) = p(d) - (d+delta)*k*A_dt*exp(-k*d) = 0
    """
    if delta <= -depth_min:
        return depth_min
    
    d = max(-1.0/k - delta, depth_min)
    d = min(d, depth_max)
    
    for _ in range(20):
        exp_kd = np.exp(-k * d)
        p = 1.0 - np.exp(-A_dt * exp_kd)
        f = p - (d + delta) * k * A_dt * exp_kd
        
        if abs(f) < 1e-13:
            break
        
        fp = k * A_dt * exp_kd * (exp_kd - 1.0 + (d + delta) * k)
        
        if abs(fp) < 1e-15:
            break
        
        d_new = d - f / fp
        d_new = float(np.clip(d_new, depth_min, depth_max))
        
        if abs(d_new - d) < 1e-14:
            d = d_new
            break
        d = d_new
    
    return float(np.clip(d, depth_min, depth_max))


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Uses Newton's method on the first-order condition for depth optimization.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    depth_min = 1e-4
    depth_max = 20.0 / k
    A_dt = A * dt

    V_next = np.zeros(2 * Q + 1)
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    q_vals = np.arange(-Q, Q + 1)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 3.0:
            for nn in range(n - 1, -1, -1):
                opt_bid[nn] = opt_bid[n]
                opt_ask[nn] = opt_ask[n]
            break

        V_curr = np.zeros(2 * Q + 1)
        penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            pen = penalty[q_idx]

            if q == Q:
                delta = V_next[q_idx - 1] - V_next[q_idx]
                best_d = _solve_depth_foc(delta, A_dt, k, depth_min, depth_max)
                exp_kd = np.exp(-k * best_d)
                p = 1.0 - np.exp(-A_dt * exp_kd)
                best_val = p * (best_d + delta) - pen + V_next[q_idx]
                V_curr[q_idx] = best_val
                opt_ask[n, q_idx] = best_d
                opt_bid[n, q_idx] = np.inf

            elif q == -Q:
                delta = V_next[q_idx + 1] - V_next[q_idx]
                best_d = _solve_depth_foc(delta, A_dt, k, depth_min, depth_max)
                exp_kd = np.exp(-k * best_d)
                p = 1.0 - np.exp(-A_dt * exp_kd)
                best_val = p * (best_d + delta) - pen + V_next[q_idx]
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_d
                opt_ask[n, q_idx] = np.inf

            else:
                delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                
                best_db = _solve_depth_foc(delta_plus, A_dt, k, depth_min, depth_max)
                exp_kd_b = np.exp(-k * best_db)
                p_b = 1.0 - np.exp(-A_dt * exp_kd_b)
                best_val_b = p_b * (best_db + delta_plus)
                
                best_da = _solve_depth_foc(delta_minus, A_dt, k, depth_min, depth_max)
                exp_kd_a = np.exp(-k * best_da)
                p_a = 1.0 - np.exp(-A_dt * exp_kd_a)
                best_val_a = p_a * (best_da + delta_minus)
                
                best_val = best_val_b + best_val_a - pen + V_next[q_idx]
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_db
                opt_ask[n, q_idx] = best_da

        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """
    Avellaneda-Stoikov style analytical quote.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    remaining = T - t
    
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)
    
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    
    bid_depths = np.clip(bid_depths, 1e-4, 20.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 20.0 / k)
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Try to compute policy if not cached and we have time
    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 15.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass
    
    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        bid_depths = np.asarray(bid_depths, dtype=np.float64)
        ask_depths = np.asarray(ask_depths, dtype=np.float64)
        return bid_depths, ask_depths
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
