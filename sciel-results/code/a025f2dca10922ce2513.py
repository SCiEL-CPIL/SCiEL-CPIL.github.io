import numpy as np
from scipy.optimize import minimize_scalar
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill
_cache = {}

def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth tables via dynamic programming.
    Uses scipy.optimize.minimize_scalar for high-precision 1D optimization.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    # Search bounds for depth
    # d is in [0, D_max]
    # p(d) = 1 - exp(-A exp(-k d) dt)
    # For large d, p(d) -> 1.
    # If d > 10/k, exp(-k d) < 4.5e-5.
    # A * 4.5e-5 * dt is very small.
    # So p(d) is very close to 1.
    # The gain d + delta_V increases linearly.
    # The marginal gain of increasing d is 1 (from the d term) minus the change in fill prob.
    # Actually, if p(d) is close to 1, g(d) ~ d + delta_V.
    # So g(d) is increasing.
    # The optimum is likely at the boundary D_max if delta_V is not very negative.
    # If delta_V is very negative, we want small d.
    D_max = 10.0 / k
    
    # Pre-compute constants for speed
    A_dt = A * dt
    
    def fill_prob(d):
        # Vectorized or scalar? minimize_scalar passes scalar.
        # return 1.0 - np.exp(-A_dt * np.exp(-k * d))
        # Use math.exp for scalar speed?
        # np.exp is fine.
        return 1.0 - np.exp(-A_dt * np.exp(-k * d))

    # We will define the objective functions inside the loop to capture delta_V
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: simple heuristic
            for nn in range(n, -1, -1):
                for qi in range(n_inv):
                    q = qi - Q
                    base = 1.0 / k
                    skew = gamma * sigma * sigma * (T - nn * dt) * q * 0.05
                    b = max(1e-6, base * (1.0 + skew))
                    a = max(1e-6, base * (1.0 - skew))
                    if q >= Q:
                        b = np.inf
                    if q <= -Q:
                        a = np.inf
                    opt_bid[nn, qi] = b
                    opt_ask[nn, qi] = a
                    V[nn, qi] = 0.0
            break

        Vn1 = V[n + 1]
        
        for qi in range(n_inv):
            q = qi - Q
            V_next_q = Vn1[qi]
            
            bid_reward = 0.0
            bid_depth = np.inf
            ask_reward = 0.0
            ask_depth = np.inf
            
            # Bid side
            if q < Q:
                delta_b = Vn1[qi + 1] - V_next_q
                
                # If delta_b is very negative, we might not want to quote.
                # But we must check if max_d p(d)(d+delta_b) > 0.
                
                def neg_obj_bid(d):
                    if d <= 0:
                        p = 1.0 - np.exp(-A_dt)
                    else:
                        p = 1.0 - np.exp(-A_dt * np.exp(-k * d))
                    return -(p * (d + delta_b))
                
                # Optimization
                # If delta_b is very large positive, optimum is at D_max.
                # If delta_b is very large negative, optimum is at 0.
                # minimize_scalar handles this.
                res = minimize_scalar(neg_obj_bid, bounds=(0.0, D_max), method='bounded', options={'xatol': 1e-6})
                d_opt = res.x
                val = -res.fun
                
                if val > 0:
                    bid_reward = val
                    bid_depth = d_opt
                else:
                    bid_reward = 0.0
                    bid_depth = np.inf
                    
            # Ask side
            if q > -Q:
                delta_a = Vn1[qi - 1] - V_next_q
                
                def neg_obj_ask(d):
                    if d <= 0:
                        p = 1.0 - np.exp(-A_dt)
                    else:
                        p = 1.0 - np.exp(-A_dt * np.exp(-k * d))
                    return -(p * (d + delta_a))
                
                res = minimize_scalar(neg_obj_ask, bounds=(0.0, D_max), method='bounded', options={'xatol': 1e-6})
                d_opt = res.x
                val = -res.fun
                
                if val > 0:
                    ask_reward = val
                    ask_depth = d_opt
                else:
                    ask_reward = 0.0
                    ask_depth = np.inf

            V[n, qi] = V_next_q - risk[qi] + bid_reward + ask_reward
            opt_bid[n, qi] = bid_depth
            opt_ask[n, qi] = ask_depth

    return opt_bid, opt_ask

def _get_dp_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key not in _cache:
        sigma, A, k, gamma, T, steps, Q = key
        opt_bid, opt_ask = _compute_dp(sigma, A, k, gamma, T, steps, Q)
        _cache[key] = (opt_bid, opt_ask)
    return _cache[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    opt_bid, opt_ask = _get_dp_tables(params)

    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].astype(np.float64)
    ask_depths = opt_ask[n, q_idx].astype(np.float64)

    # Safety: ensure valid values
    min_valid = -3.0 / k + 1e-6
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)

    # Boundary: no bid at Q, no ask at -Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
