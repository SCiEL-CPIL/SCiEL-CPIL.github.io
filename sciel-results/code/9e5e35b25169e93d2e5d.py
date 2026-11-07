import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0
_cache = {}


def _vectorized_optimize_batch(A_dt, k, deltas, D_max, n_grid=2000, n_refine=8):
    """
    Vectorized 1D optimization of g(d) = p(d) * (d + delta) over d in [0, D_max]
    for a batch of delta values.
    p(d) = 1 - exp(-A_dt * exp(-k*d)).
    Returns: (d_opts, vals) arrays of shape (len(deltas),).
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)

    d_grid = np.linspace(0.0, D_max, n_grid)
    p_grid = 1.0 - np.exp(-A_dt * np.exp(-k * d_grid))
    g_grid = p_grid[None, :] * (d_grid[None, :] + deltas[:, None])
    idx_best = np.argmax(g_grid, axis=1)

    d_lo = np.zeros(n, dtype=np.float64)
    d_hi = np.zeros(n, dtype=np.float64)

    left_mask = (idx_best == 0)
    d_lo[left_mask] = 0.0
    d_hi[left_mask] = d_grid[1] if n_grid > 1 else D_max

    right_mask = (idx_best == n_grid - 1)
    d_lo[right_mask] = d_grid[-2] if n_grid > 1 else 0.0
    d_hi[right_mask] = D_max

    int_mask = ~(left_mask | right_mask)
    d_lo[int_mask] = d_grid[idx_best[int_mask] - 1]
    d_hi[int_mask] = d_grid[idx_best[int_mask] + 1]

    for _ in range(n_refine):
        d_mid = 0.5 * (d_lo + d_hi)
        exp_kd = np.exp(-k * d_mid)
        u = A_dt * exp_kd
        p = 1.0 - np.exp(-u)
        dp_dd = -k * u * np.exp(-u)
        gp = dp_dd * (d_mid + deltas) + p
        d_lo = np.where(gp > 0, d_mid, d_lo)
        d_hi = np.where(gp > 0, d_hi, d_mid)

    d_opt = 0.5 * (d_lo + d_hi)
    p_opt = 1.0 - np.exp(-A_dt * np.exp(-k * d_opt))
    val_opt = p_opt * (d_opt + deltas)

    p_zero = 1.0 - np.exp(-A_dt)
    val_zero = p_zero * deltas

    p_dmax = 1.0 - np.exp(-A_dt * np.exp(-k * D_max))
    val_dmax = p_dmax * (D_max + deltas)

    best_val = np.maximum.reduce([np.zeros(n), val_opt, val_zero, val_dmax])
    d_result = np.where(val_opt >= val_zero, d_opt, 0.0)
    d_result = np.where(val_dmax >= np.maximum(val_opt, val_zero), D_max, d_result)
    val_result = np.where(best_val <= 0, 0.0, best_val)
    return d_result, val_result


def _fast_optimize_scalar(A_dt, k, delta, D_max, n_refine=20):
    """
    Fast analytic first step + bisection polish for a single delta.
    Returns (d_opt, val).
    """
    # g(d) = p(d) * (d + delta), p(d) = 1 - exp(-A_dt * exp(-k*d))
    # g'(d) = -k * u * exp(-u) * (d + delta) + (1 - exp(-u)), u = A_dt * exp(-k*d)
    # g'(0) = -k * A_dt * exp(-A_dt) * delta + (1 - exp(-A_dt))
    # If g'(0) <= 0, optimum is at d=0 (boundary)
    u0 = A_dt
    if u0 < 1e-12:
        return 0.0, 0.0
    p0 = 1.0 - np.exp(-u0)
    gp0 = -k * u0 * np.exp(-u0) * delta + p0
    if gp0 <= 0:
        return 0.0, p0 * delta

    # Otherwise, find root of g'(d)=0 in (0, D_max) via bisection
    d_lo, d_hi = 0.0, D_max
    for _ in range(n_refine):
        d_mid = 0.5 * (d_lo + d_hi)
        u = A_dt * np.exp(-k * d_mid)
        p = 1.0 - np.exp(-u)
        gp = -k * u * np.exp(-u) * (d_mid + delta) + p
        if gp > 0:
            d_lo = d_mid
        else:
            d_hi = d_mid
    d_opt = 0.5 * (d_lo + d_hi)
    p_opt = 1.0 - np.exp(-A_dt * np.exp(-k * d_opt))
    val_opt = p_opt * (d_opt + delta)
    # Compare with d=0
    if val_opt >= p0 * delta:
        return d_opt, val_opt
    else:
        return 0.0, p0 * delta


def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth tables via dynamic programming.
    Uses time-homogeneity: policy depends only on m = steps - n (remaining steps).
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt
    D_max = 8.0 / k
    A_dt = A * dt

    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    V_next = np.zeros(n_inv, dtype=np.float64)

    for m in range(1, steps + 1):
        if time.time() > _DEADLINE:
            # Fallback heuristic for remaining m
            for mm in range(m, steps + 1):
                n_idx = steps - mm
                for qi in range(n_inv):
                    q = qi - Q
                    base = 1.0 / k
                    skew = gamma * sigma * sigma * (mm * dt) * q * 0.05
                    b = max(1e-6, base * (1.0 + skew))
                    a = max(1e-6, base * (1.0 - skew))
                    if q >= Q:
                        b = np.inf
                    if q <= -Q:
                        a = np.inf
                    opt_bid[n_idx, qi] = b
                    opt_ask[n_idx, qi] = a
            return opt_bid, opt_ask

        n_idx = steps - m  # absolute step index for this m

        # Bid deltas: delta_b[qi] = V_next[qi+1] - V_next[qi] for qi=0..2Q-1
        delta_b_full = V_next[1:] - V_next[:2 * Q]  # shape (2Q,)
        # Ask deltas: delta_a[qi] = V_next[qi-1] - V_next[qi] for qi=1..2Q
        delta_a_full = V_next[:2 * Q] - V_next[1:]  # shape (2Q,)

        bid_depths_full = np.full(2 * Q, np.inf, dtype=np.float64)
        bid_rewards_full = np.zeros(2 * Q, dtype=np.float64)
        ask_depths_full = np.full(2 * Q, np.inf, dtype=np.float64)
        ask_rewards_full = np.zeros(2 * Q, dtype=np.float64)

        # Fast scalar optimization for each inventory level
        for i in range(2 * Q):
            d_opt, val = _fast_optimize_scalar(A_dt, k, delta_b_full[i], D_max)
            bid_depths_full[i] = d_opt
            bid_rewards_full[i] = val
            d_opt, val = _fast_optimize_scalar(A_dt, k, delta_a_full[i], D_max)
            ask_depths_full[i] = d_opt
            ask_rewards_full[i] = val

        opt_bid[n_idx, 0:2 * Q] = bid_depths_full
        opt_bid[n_idx, 2 * Q] = np.inf
        opt_ask[n_idx, 0] = np.inf
        opt_ask[n_idx, 1:2 * Q + 1] = ask_depths_full

        total_reward = np.zeros(n_inv, dtype=np.float64)
        total_reward[0:2 * Q] += bid_rewards_full
        total_reward[1:2 * Q + 1] += ask_rewards_full

        V_curr = V_next - risk + total_reward
        V_next = V_curr

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

    min_valid = -3.0 / k + 1e-6
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)

    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
