import numpy as np
import time

_start_time = time.time()
_DEADLINE = 315  # safety margin before the 330s kill
_cache = {}


def _opt_depth(delta, A, k, dt):
    """Return (opt_depth, max_value) for max over d>0 of p(d)*(d+delta),
    p(d) = 1 - exp(-A*exp(-k*d)*dt)."""
    lam = A * dt
    if delta <= 0:
        return 0.0, 0.0
    # Interior FOC: d*exp(-k*d) = (delta/lambda)*(1 - exp(-lambda*exp(-k*d)))
    # Closed-form (Lambert W) for the delta/lambda >> 1 regime (lambda*dt tiny here).
    r = delta / lam
    x = r / np.e
    W = np.expm1(np.log(x) - np.log1p(np.log(x)))  # W(x)
    d0 = (np.log(r) - np.log(W)) / k
    if d0 <= 1e-9 or not np.isfinite(d0):
        return 0.0, 0.0

    # Newton polish on f(d)=p(d)*(d+delta)-V, a few cheap iterations.
    d = d0
    V = 0.0
    for _ in range(3):
        e = np.exp(-k * d)
        p = 1.0 - np.exp(-lam * e)
        V = p * (d + delta)
        # f'(d) = p'*(d+delta) + p
        # p' = -lam*e*(-k)*exp(-lam*e) = k*lam*e*exp(-lam*e)
        fp = (k * lam * e * np.exp(-lam * e)) * (d + delta) + p
        if abs(fp) < 1e-14:
            break
        step = V / fp
        d -= step
        if d <= 1e-9:
            d = 1e-9
            break
        if abs(step) < 1e-12 * d:
            break
    e = np.exp(-k * d)
    p = 1.0 - np.exp(-lam * e)
    V = p * (d + delta)
    if V <= 0 or not np.isfinite(V):
        return 0.0, 0.0
    return d, V


def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    n_inv = 2 * Q + 1
    lam = A * dt

    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt
    q_idx = np.arange(n_inv)

    V = np.zeros(n_inv, dtype=np.float64)  # V[n+1]
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback heuristic for remaining steps
            for nn in range(n, -1, -1):
                base = 1.0 / k
                for qi in range(n_inv):
                    q = qi - Q
                    skew = gamma * sigma * sigma * (T - nn * dt) * q * 0.03
                    b = max(1e-6, base * (1.0 + skew))
                    a = max(1e-6, base * (1.0 - skew))
                    if q >= Q:
                        b = np.inf
                    if q <= -Q:
                        a = np.inf
                    opt_bid[nn, qi] = b
                    opt_ask[nn, qi] = a
                    V[qi] = 0.0
            break

        Vn1 = V  # V[n+1]
        Vnew = np.empty(n_inv, dtype=np.float64)

        # Bid side: fill moves q -> q+1 (only if q < Q)
        delta_b = np.empty(n_inv, dtype=np.float64)
        delta_b[:n_inv - 1] = Vn1[1:] - Vn1[:n_inv - 1]
        delta_b[n_inv - 1] = 0.0
        # Ask side: fill moves q -> q-1 (only if q > -Q)
        delta_a = np.empty(n_inv, dtype=np.float64)
        delta_a[1:] = Vn1[:n_inv - 1] - Vn1[1:]
        delta_a[0] = 0.0

        bid_d = np.full(n_inv, np.inf, dtype=np.float64)
        ask_d = np.full(n_inv, np.inf, dtype=np.float64)
        bid_val = np.zeros(n_inv, dtype=np.float64)
        ask_val = np.zeros(n_inv, dtype=np.float64)

        for qi in range(n_inv):
            q = qi - Q
            if q < Q:
                d, v = _opt_depth(delta_b[qi], A, k, dt)
                bid_d[qi] = d
                bid_val[qi] = v
            if q > -Q:
                d, v = _opt_depth(delta_a[qi], A, k, dt)
                ask_d[qi] = d
                ask_val[qi] = v

        Vnew = Vn1 - risk + bid_val + ask_val
        opt_bid[n, :] = bid_d
        opt_ask[n, :] = ask_d
        V = Vnew

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
    k = params['k']
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
