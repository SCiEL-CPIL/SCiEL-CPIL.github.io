import numpy as np
from scipy.optimize import minimize_scalar
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill
_cache = {}

def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid/ask depth tables via dynamic programming with exact 1-D maximization."""
    dt = T / steps
    n_inv = 2 * Q + 1

    # V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    # Precompute fill probability as a function of depth for vectorized use
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # We want to maximize: p(d) * (d + delta_V)
    # where delta_V is the continuation value difference

    d_max = 12.0 / k  # depths beyond this give negligible fill prob

    def max_one_side(delta_V, can_fill):
        """Maximize p(d)*(d + delta_V) over d in [0, d_max]. Return (max_val, d_opt)."""
        if not can_fill:
            return 0.0, np.inf

        def neg_obj(d):
            if d <= 0:
                return 0.0
            p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
            return -(p * (d + delta_V))

        # Use bounded scalar minimization
        res = minimize_scalar(neg_obj, bounds=(1e-8, d_max), method='bounded',
                              options={'xatol': 1e-9})
        d_opt = max(1e-8, res.x)
        val = -res.fun if np.isfinite(res.fun) else 0.0
        if val < 0:
            val = 0.0
            d_opt = np.inf
        return val, d_opt

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: fill remaining with a simple heuristic
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

        for qi in range(n_inv):
            q = qi - Q
            Vn1_q = V[n + 1, qi]

            # Bid side: fill moves inventory from q to q+1
            if q < Q:
                delta_b = V[n + 1, qi + 1] - Vn1_q
                bid_val, bid_d = max_one_side(delta_b, True)
            else:
                bid_val = 0.0
                bid_d = np.inf

            # Ask side: fill moves inventory from q to q-1
            if q > -Q:
                delta_a = V[n + 1, qi - 1] - Vn1_q
                ask_val, ask_d = max_one_side(delta_a, True)
            else:
                ask_val = 0.0
                ask_d = np.inf

            V[n, qi] = Vn1_q - risk[qi] + bid_val + ask_val
            opt_bid[n, qi] = bid_d
            opt_ask[n, qi] = ask_d

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
