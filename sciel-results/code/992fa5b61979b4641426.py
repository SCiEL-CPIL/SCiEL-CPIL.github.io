import numpy as np
import time

_start_time = time.time()
_DEADLINE = 315  # safety margin before 330s kill
_cache = {}


def _solve_side(delta_V, A, k, dt, d_lo, d_hi, n_grid, n_refine_iters):
    """
    Maximize g(d) = p(d)*(d + delta_V) over d in [d_lo, d_hi].
    p(d) = 1 - exp(-A*exp(-k*d)*dt).
    Returns (max_val, d_opt). Uses coarse grid + local refinement + bisection polish.
    """
    # Coarse grid
    d_grid = np.linspace(d_lo, d_hi, n_grid)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
    g_grid = p_grid * (d_grid + delta_V)

    best_idx = int(np.argmax(g_grid))
    best_val = g_grid[best_idx]
    if best_val <= 0:
        return 0.0, np.inf

    # Local refinement around best_idx
    lo = max(0, best_idx - 3)
    hi = min(n_grid - 1, best_idx + 3)
    dl = d_grid[lo]
    dh = d_grid[hi]

    # Bisection on g'(d) = 0. g'(d) = p'(d)*(d+delta) + p(d).
    # p'(d) = A*k*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)
    def gprime(d):
        e = np.exp(-k * d)
        p = 1.0 - np.exp(-A * e * dt)
        pp = A * k * e * dt * np.exp(-A * e * dt)
        return pp * (d + delta_V) + p

    # Check sign at endpoints to bracket the root of g' (where g' crosses 0 from + to -)
    gp_lo = gprime(dl)
    gp_hi = gprime(dh)

    d_opt = d_grid[best_idx]
    if gp_lo > 0 and gp_hi < 0:
        # Bisection for high precision
        a, b = dl, dh
        for _ in range(n_refine_iters):
            m = 0.5 * (a + b)
            gm = gprime(m)
            if gm > 0:
                a = m
            else:
                b = m
        d_opt = 0.5 * (a + b)
    elif gp_lo > 0:
        # root is beyond dh, take dh
        d_opt = dh
    elif gp_hi < 0:
        # root is before dl, take dl
        d_opt = dl
    else:
        # both same sign, fall back to grid argmax
        d_opt = d_grid[best_idx]

    d_opt = max(d_lo, min(d_hi, d_opt))
    e = np.exp(-k * d_opt)
    p_opt = 1.0 - np.exp(-A * e * dt)
    val = p_opt * (d_opt + delta_V)
    if not np.isfinite(val) or val < 0:
        val = 0.0
        d_opt = np.inf
    return val, d_opt


def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    n_inv = 2 * Q + 1

    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    d_lo = 1e-8
    d_hi = 20.0 / k
    n_grid = 400
    n_refine_iters = 50

    # Precompute grid (shared across all calls)
    d_grid = np.linspace(d_lo, d_hi, n_grid)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback heuristic
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

        # Vectorized coarse evaluation for all inventory levels
        # delta_b[i] = V[n+1, i+1] - V[n+1, i] for i in 0..n_inv-2 (q < Q)
        # delta_a[i] = V[n+1, i-1] - V[n+1, i] for i in 1..n_inv-1 (q > -Q)
        Vn1 = V[n + 1]
        # bid: fill q->q+1, valid for q<Q i.e. qi < n_inv-1
        delta_b = np.empty(n_inv, dtype=np.float64)
        delta_b[:n_inv - 1] = Vn1[1:n_inv] - Vn1[0:n_inv - 1]
        delta_b[n_inv - 1] = 0.0
        # ask: fill q->q-1, valid for q>-Q i.e. qi > 0
        delta_a = np.empty(n_inv, dtype=np.float64)
        delta_a[1:n_inv] = Vn1[0:n_inv - 1] - Vn1[1:n_inv]
        delta_a[0] = 0.0

        # Coarse g for each side
        g_b = p_grid[None, :] * (d_grid[None, :] + delta_b[:, None])
        g_a = p_grid[None, :] * (d_grid[None, :] + delta_a[:, None])

        bi_idx = np.argmax(g_b, axis=1)
        ai_idx = np.argmax(g_a, axis=1)
        bi_val = g_b[np.arange(n_inv), bi_idx]
        ai_val = g_a[np.arange(n_inv), ai_idx]

        # Local bisection refinement per inventory level
        for qi in range(n_inv):
            q = qi - Q
            # Bid
            if q < Q and bi_val[qi] > 0:
                lo = max(0, bi_idx[qi] - 3)
                hi = min(n_grid - 1, bi_idx[qi] + 3)
                dl = d_grid[lo]
                dh = d_grid[hi]
                dV = delta_b[qi]

                def gprime_b(d, dV=dV):
                    e = np.exp(-k * d)
                    p = 1.0 - np.exp(-A * e * dt)
                    pp = A * k * e * dt * np.exp(-A * e * dt)
                    return pp * (d + dV) + p

                gp_lo = gprime_b(dl)
                gp_hi = gprime_b(dh)
                d_opt = d_grid[bi_idx[qi]]
                if gp_lo > 0 and gp_hi < 0:
                    a, b = dl, dh
                    for _ in range(n_refine_iters):
                        m = 0.5 * (a + b)
                        if gprime_b(m) > 0:
                            a = m
                        else:
                            b = m
                    d_opt = 0.5 * (a + b)
                elif gp_lo > 0:
                    d_opt = dh
                elif gp_hi < 0:
                    d_opt = dl
                d_opt = max(d_lo, min(d_hi, d_opt))
                e = np.exp(-k * d_opt)
                p_opt = 1.0 - np.exp(-A * e * dt)
                val = p_opt * (d_opt + dV)
                if not np.isfinite(val) or val < 0:
                    val = 0.0
                    d_opt = np.inf
                bi_val[qi] = val
                opt_bid[n, qi] = d_opt
            else:
                bi_val[qi] = 0.0
                opt_bid[n, qi] = np.inf

            # Ask
            if q > -Q and ai_val[qi] > 0:
                lo = max(0, ai_idx[qi] - 3)
                hi = min(n_grid - 1, ai_idx[qi] + 3)
                dl = d_grid[lo]
                dh = d_grid[hi]
                dV = delta_a[qi]

                def gprime_a(d, dV=dV):
                    e = np.exp(-k * d)
                    p = 1.0 - np.exp(-A * e * dt)
                    pp = A * k * e * dt * np.exp(-A * e * dt)
                    return pp * (d + dV) + p

                gp_lo = gprime_a(dl)
                gp_hi = gprime_a(dh)
                d_opt = d_grid[ai_idx[qi]]
                if gp_lo > 0 and gp_hi < 0:
                    a, b = dl, dh
                    for _ in range(n_refine_iters):
                        m = 0.5 * (a + b)
                        if gprime_a(m) > 0:
                            a = m
                        else:
                            b = m
                    d_opt = 0.5 * (a + b)
                elif gp_lo > 0:
                    d_opt = dh
                elif gp_hi < 0:
                    d_opt = dl
                d_opt = max(d_lo, min(d_hi, d_opt))
                e = np.exp(-k * d_opt)
                p_opt = 1.0 - np.exp(-A * e * dt)
                val = p_opt * (d_opt + dV)
                if not np.isfinite(val) or val < 0:
                    val = 0.0
                    d_opt = np.inf
                ai_val[qi] = val
                opt_ask[n, qi] = d_opt
            else:
                ai_val[qi] = 0.0
                opt_ask[n, qi] = np.inf

            V[n, qi] = Vn1[qi] - risk[qi] + bi_val[qi] + ai_val[qi]

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
