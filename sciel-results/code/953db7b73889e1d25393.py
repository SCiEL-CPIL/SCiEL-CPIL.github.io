import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill
_cache = {}

def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid/ask depth tables via dynamic programming with vectorized 1-D maximization."""
    dt = T / steps
    n_inv = 2 * Q + 1

    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    # Depth bounds for Newton's method
    d_max = 12.0 / k

    def max_one_side_vectorized(deltas):
        """
        Vectorized Newton's method for each delta in deltas array.
        Maximize g(d) = p(d)*(d+delta) where p(d) = 1 - exp(-A*exp(-k*d)*dt)
        Returns (vals, d_opts) arrays.
        """
        n = len(deltas)
        d = np.full(n, 1.0 / k, dtype=np.float64)  # initial guess
        
        # Check boundary: if g'(0) <= 0, don't quote
        # g'(d) = k*p*(d+delta) - exp(-lam)  (from FOC)
        lam_0 = A * dt  # at d=0
        p_0 = 1.0 - np.exp(-lam_0)
        gprime_0 = k * p_0 * (0.0 + deltas) - np.exp(-lam_0)
        no_quote = gprime_0 <= 0
        
        # Newton iterations for those that should quote
        active = ~no_quote
        d_active = d[active]
        delta_active = deltas[active]
        
        lo = 1e-10
        hi = d_max
        
        for _ in range(30):
            lam = A * np.exp(-k * d_active) * dt
            exp_lam = np.exp(-lam)
            p = 1.0 - exp_lam
            
            # f(d) = k*p*(d+delta) - exp(-lam)
            fv = k * p * (d_active + delta_active) - exp_lam
            
            # f'(d) = k*p - k*lam*exp_lam*(k*(d+delta) + 1)
            fpv = k * p - k * lam * exp_lam * (k * (d_active + delta_active) + 1.0)
            
            # Check convergence
            converged = np.abs(fv) < 1e-12
            not_converged = ~converged
            
            if not np.any(not_converged):
                break
            
            d_active[not_converged] = np.where(
                np.abs(fpv[not_converged]) > 1e-15,
                d_active[not_converged] - fv[not_converged] / fpv[not_converged],
                (lo + hi) / 2.0
            )
            
            # Bisection safeguard
            d_active[not_converged] = np.clip(d_active[not_converged], lo, hi)
        
        d[active] = d_active
        d = np.clip(d, lo, d_max)
        
        # Compute final values
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        vals = p * (d + deltas)
        
        # For no_quote, set to 0 and inf
        vals[no_quote] = 0.0
        d[no_quote] = np.inf
        
        # Safety: if val < 0, no quote
        bad = vals < 0
        vals[bad] = 0.0
        d[bad] = np.inf
        
        return vals, d

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

        # Bid side: fill moves inventory from q to q+1
        # For q in [-Q, Q-1], delta_b[i] = V[n+1, i+1] - V[n+1, i]
        # For q = Q, no bid
        deltas_b = np.full(n_inv, 0.0)
        deltas_b[:Q] = V[n + 1, 1:Q+1] - V[n + 1, 0:Q]  # q=-Q..Q-1
        # q=Q: no fill possible
        
        bid_vals, bid_ds = max_one_side_vectorized(deltas_b)
        bid_vals[Q] = 0.0
        bid_ds[Q] = np.inf

        # Ask side: fill moves inventory from q to q-1
        # For q in [-Q+1, Q], delta_a[i] = V[n+1, i-1] - V[n+1, i]
        # For q = -Q, no ask
        deltas_a = np.full(n_inv, 0.0)
        deltas_a[Q+1:] = V[n + 1, 0:Q] - V[n + 1, 1:Q+1]  # q=-Q+1..Q
        # q=-Q: no fill possible
        
        ask_vals, ask_ds = max_one_side_vectorized(deltas_a)
        ask_vals[0] = 0.0
        ask_ds[0] = np.inf

        V[n, :] = V[n + 1, :] - risk + bid_vals + ask_vals
        opt_bid[n, :] = bid_ds
        opt_ask[n, :] = ask_ds

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
