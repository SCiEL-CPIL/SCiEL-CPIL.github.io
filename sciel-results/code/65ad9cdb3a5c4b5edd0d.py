"""Optimal market making with inventory risk via dynamic programming.

The model is mid-invariant (rewards are depth-based), so the optimal policy
depends only on (time index, inventory).  We solve the finite-horizon Bellman
recursion with a concave piecewise-linear value function V(q) parameterized by
marginal-value breakpoints, and optimize bid/ask depths in closed form.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 4.0

_cache = {}


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """Return (bid_table, ask_table) arrays of shape (steps, 2Q+1)."""
    dt = T / steps
    nQ = 2 * Q + 1
    qidx = np.arange(-Q, Q + 1)          # q = -Q..Q
    qi = np.arange(nQ)

    # Precompute fill-probability / log tables over a depth grid.
    # depth in [dmin, dmax]; dmin = -3/k (floor), dmax large.
    dmin = -3.0 / k
    dmax = 8.0 / k + 6.0
    nD = 20000
    depths = np.linspace(dmin, dmax, nD)
    # p = 1 - exp(-A exp(-k d) dt);  log-odds for optimization
    # We need, for a given target fill probability p* in (0,1):
    #   exp(-k d) = -log(1-p*) / (A dt)  =>  d = -log( -log(1-p*)/(A dt) )/k
    # valid when -log(1-p*) in (0, A dt).  p* range: (0, 1-exp(-A dt)).
    lam_max = A * dt                      # max of A exp(-k d) dt as d->-inf
    p_max = 1.0 - np.exp(-lam_max)        # max fill prob (d -> -inf)

    # Value function: concave PWL.  Marginal values m[i] = V(q_i) - V(q_{i-1}),
    # non-increasing in i.  Represent by array m of length nQ (m[0] = V(-Q)-V(-Q-1)
    # is a free boundary slope; we keep it finite).  Initialize to zero.
    m = np.zeros(nQ)

    for n in range(steps - 1, -1, -1):
        risk = 0.5 * gamma * sigma * sigma * dt
        # For each q, choose depths to maximize:
        #   p_b*(b + V(q+1)-V(q)) + p_a*(a + V(q-1)-V(q)) - risk*q^2
        # with q+1 capped at Q (no bid at q=Q), q-1 capped at -Q (no ask at q=-Q).
        # Marginal gain of a bid fill: b + (V(q+1)-V(q)) = b + m[q+1]
        # Marginal gain of an ask fill: a + (V(q-1)-V(q)) = a - m[q]
        # where m[q] = V(q)-V(q-1).  (index: m[i] for transition q_{i-1}->q_i)
        # For q = i in qidx, m_bid = m[i+1] if i+1<nQ else 0 (no bid at Q)
        # m_ask = -m[i] (since V(q-1)-V(q) = -m[i])

        bid_gain = np.zeros(nQ)
        ask_gain = np.zeros(nQ)
        # m[i+1] for i in 0..nQ-2
        bid_gain[:nQ - 1] = m[1:]
        # at i = nQ-1 (q=Q): no bid -> gain irrelevant, depth = inf
        # m[i] for i in 1..nQ-1
        ask_gain[1:] = -m[1:]
        # at i = 0 (q=-Q): no ask

        # Optimal depth for a side with marginal gain g:
        # maximize p(d)*(d + g) over d >= dmin, p(d)=1-exp(-A e^{-k d} dt)
        # Let u = A e^{-k d} dt  (u in (0, lam_max]).  d = -log(u/(A dt))/k.
        # f(u) = (1-e^{-u})*(-log(u/(A dt))/k + g)
        # Optimize over u.  This is 1-D; do a coarse grid + refine.
        # We compute optimal depth for each distinct g value via vectorized search.
        bid_depth = np.full(nQ, np.inf)
        ask_depth = np.full(nQ, np.inf)

        # Build candidate u grid (log-spaced in (0, lam_max]).
        u_grid = np.geomspace(1e-6, lam_max, 400)
        u_grid = np.clip(u_grid, 1e-9, lam_max)

        def opt_depth(g):
            # g array; returns optimal depth array (>= dmin), or inf if no fill.
            # objective over u_grid
            d_u = -np.log(u_grid / (A * dt)) / k
            p_u = 1.0 - np.exp(-u_grid)
            obj = p_u * (d_u + g[None, :])          # (nU, nQ)
            # also consider d = dmin (u = A dt e^{-k dmin}, possibly > lam_max)
            d_min = dmin
            u_min = A * np.exp(-k * d_min) * dt
            p_min = 1.0 - np.exp(-min(u_min, lam_max))
            obj_min = p_min * (d_min + g)           # (nQ,)
            obj = np.vstack([obj, obj_min[None, :]])
            idx = np.argmax(obj, axis=0)
            # map idx back to depth
            best = np.empty(nQ)
            in_grid = idx < len(u_grid)
            best[in_grid] = d_u[idx[in_grid], np.where(in_grid)[0]]
            best[~in_grid] = d_min
            # enforce floor
            best = np.maximum(best, dmin)
            # if best objective <= 0 and a no-quote (inf) is allowed, prefer no quote?
            # No quote gives 0 from that side.  If max obj < 0, quoting loses; set inf.
            # but we must still have a valid depth; using inf means no fill that side.
            # We'll set to inf where best objective < 0.
            obj_best = np.max(obj, axis=0)
            best[obj_best < 0.0] = np.inf
            return best

        # Only compute for states where quoting is possible.
        # bid possible for i in 0..nQ-2 ; ask possible for i in 1..nQ-1
        bid_depth[:nQ - 1] = opt_depth(bid_gain[:nQ - 1])
        ask_depth[1:] = opt_depth(ask_gain[1:])

        # Update value function: V(q) = max_depths [ ... ] - risk*q^2
        # We need V_new(q).  Compute the max objective per q (sum of both sides'
        # best objectives) minus risk.
        # Recompute per-side best objective for the chosen depth.
        def side_obj(depth, g):
            # depth may be inf
            valid = np.isfinite(depth)
            u = A * np.exp(-k * depth[valid]) * dt
            u = np.clip(u, 0.0, lam_max)
            p = 1.0 - np.exp(-u)
            o = np.zeros(nQ)
            o[valid] = p * (depth[valid] + g[valid])
            return o

        V_new = (side_obj(bid_depth, bid_gain) +
                 side_obj(ask_depth, ask_gain) -
                 risk * qidx * qidx)
        # Convert V to marginal slopes m: m[i] = V(q_i)-V(q_{i-1}), i>=1; m[0] free.
        m_new = np.empty(nQ)
        m_new[1:] = np.diff(V_new)
        # m[0]: set to a continuation of the slope (m[1]) to keep concavity-ish.
        m_new[0] = m_new[1]
        # Enforce concavity (non-increasing m) via isotonic (pool adjacent violators)
        # to keep the PWL concave and stable.
        m = _isotonic_decreasing(m_new)

    # Tables: rows = step n, cols = q index (0..nQ-1)
    # We only stored the final m; we need per-step tables.  Re-run storing tables.
    # (Redo with storage to keep code simple.)
    return _solve_with_tables(sigma, A, k, gamma, T, steps, Q)


def _isotonic_decreasing(x):
    """Pool-adjacent-violators to make x non-increasing (concave PWL)."""
    x = x.astype(float).copy()
    n = len(x)
    # standard PAV for non-increasing
    blocks = []  # list of [sum, count, value]
    for v in x:
        blocks.append([v, 1, v])
        while len(blocks) > 1 and blocks[-2][2] < blocks[-1][2]:
            # merge (need non-increasing: prev >= next)
            s = blocks[-2][0] + blocks[-1][0]
            c = blocks[-2][1] + blocks[-1][1]
            blocks.pop()
            blocks.pop()
            blocks.append([s, c, s / c])
    out = np.empty(n)
    pos = 0
    for s, c, val in blocks:
        out[pos:pos + c] = val
        pos += c
    return out


def _solve_with_tables(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    nQ = 2 * Q + 1
    qidx = np.arange(-Q, Q + 1)
    dmin = -3.0 / k
    lam_max = A * dt
    risk = 0.5 * gamma * sigma * sigma * dt

    bid_table = np.empty((steps, nQ))
    ask_table = np.empty((steps, nQ))

    m = np.zeros(nQ)
    u_grid = np.geomspace(1e-6, lam_max, 400)
    u_grid = np.clip(u_grid, 1e-9, lam_max)
    d_u = -np.log(u_grid / (A * dt)) / k
    p_u = 1.0 - np.exp(-u_grid)
    u_min = A * np.exp(-k * dmin) * dt
    p_min = 1.0 - np.exp(-min(u_min, lam_max))

    for n in range(steps - 1, -1, -1):
        bid_gain = np.zeros(nQ)
        ask_gain = np.zeros(nQ)
        bid_gain[:nQ - 1] = m[1:]
        ask_gain[1:] = -m[1:]

        def opt_depth(g, side):
            # side: 'bid' or 'ask' -> which indices are valid
            valid_idx = np.arange(nQ)
            obj = p_u[None, :] * (d_u[None, :] + g[:, None])  # (nQ, nU)
            obj_min = p_min * (dmin + g)
            obj = np.concatenate([obj, obj_min[:, None]], axis=1)
            idx = np.argmax(obj, axis=1)
            best = np.empty(nQ)
            in_grid = idx < len(u_grid)
            best[in_grid] = d_u[in_grid, idx[in_grid]]
            best[~in_grid] = dmin
            best = np.maximum(best, dmin)
            obj_best = np.max(obj, axis=1)
            best[obj_best < 0.0] = np.inf
            return best

        bd = opt_depth(bid_gain, 'bid')
        ad = opt_depth(ask_gain, 'ask')
        # disable at boundaries
        bd[nQ - 1] = np.inf   # q = Q: no bid
        ad[0] = np.inf        # q = -Q: no ask

        bid_table[n] = bd
        ask_table[n] = ad

        def side_obj(depth, g):
            valid = np.isfinite(depth)
            o = np.zeros(nQ)
            if np.any(valid):
                u = A * np.exp(-k * depth[valid]) * dt
                u = np.clip(u, 0.0, lam_max)
                p = 1.0 - np.exp(-u)
                o[valid] = p * (depth[valid] + g[valid])
            return o

        V_new = (side_obj(bd, bid_gain) + side_obj(ad, ask_gain) -
                 risk * qidx * qidx)
        m_new = np.empty(nQ)
        m_new[1:] = np.diff(V_new)
        m_new[0] = m_new[1]
        m = _isotonic_decreasing(m_new)

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _cache:
        if _time_left() < 6.0:
            # Not enough time to solve; fall back to a cheap heuristic.
            dmin = -3.0 / k
            d = np.full(len(inventory), 1.0 / k)
            return d, d.copy()
        _cache[key] = _solve_with_tables(sigma, A, k, gamma, T, steps, Q)

    bid_table, ask_table = _cache[key]
    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)
    q = np.clip(inventory, -Q, Q)
    idx = (q + Q).astype(int)
    bid = bid_table[n, idx]
    ask = ask_table[n, idx]
    # enforce floor and validity
    dmin = -3.0 / k
    bid = np.where(np.isfinite(bid), np.maximum(bid, dmin), np.inf)
    ask = np.where(np.isfinite(ask), np.maximum(ask, dmin), np.inf)
    return bid, ask
# EVOLVE-BLOCK-END
