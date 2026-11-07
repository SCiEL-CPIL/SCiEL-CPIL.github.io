"""Optimal market making with inventory risk: precomputed HJB backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # safety margin before 300s limit

# Precomputed HJB tables keyed by (sigma, A, k, gamma, T, steps, Q)
_HJB_CACHE = {}


def _parabolic_refine(f, d, i):
    """Parabolic refinement around grid index i (clamped)."""
    n = len(d)
    if i <= 0 or i >= n - 1:
        return d[i]
    f0, f1, f2 = f[i - 1], f[i], f[i + 1]
    denom = (f0 - 2.0 * f1 + f2)
    if abs(denom) < 1e-18:
        return d[i]
    delta = 0.5 * (f0 - f2) / denom
    if delta < -1.0:
        delta = -1.0
    elif delta > 1.0:
        delta = 1.0
    return d[i - 1] + delta * (d[i] - d[i - 1])


def _solve_hjb(sigma, A, k, gamma, T, steps, Q, deadline):
    """Solve the discrete HJB via backward induction. Returns (bid_table, ask_table).

    bid_table[n, q] = optimal bid depth at step n, inventory q (q in [-Q, Q]).
    ask_table[n, q] = optimal ask depth at step n, inventory q.
    """
    nQ = 2 * Q + 1
    # Value function V[n, q] for n in [0, steps], q in [-Q, Q]
    V = np.zeros((steps + 1, nQ), dtype=np.float64)
    bid_table = np.zeros((steps, nQ), dtype=np.float64)
    ask_table = np.zeros((steps, nQ), dtype=np.float64)

    dt = T / steps
    inv_k = 1.0 / k

    # Depth grid for optimization (include 0 and 1/k)
    d_max = max(8.0 * inv_k, 4.0)
    N_D = 1200
    depth_grid = np.linspace(0.0, d_max, N_D)
    # Precompute p(d) = 1 - exp(-A exp(-k d) dt) on the grid
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    # Precompute d * p(d)
    dp_grid = depth_grid * p_grid

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with 1/k
            bid_table[n:] = inv_k
            ask_table[n:] = inv_k
            break
        Vn = V[n + 1]
        Vn_qm1 = np.concatenate(([0.0], Vn[:-1]))   # V(n+1, q-1), q-1 >= -Q
        Vn_qp1 = np.concatenate((Vn[1:], [0.0]))    # V(n+1, q+1), q+1 <= Q

        # Penalty for inventory q at this step
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        penalty = 0.5 * gamma * sigma * sigma * q_vals * q_vals * dt

        bid_row = np.zeros(nQ, dtype=np.float64)
        ask_row = np.zeros(nQ, dtype=np.float64)

        # --- Optimize bid depth for each q ---
        # For q < Q: maximize over d: d*p(d) + p(d)*Vn_qm1[q] + (1-p(d))*Vn[q]
        #   = Vn[q] + p(d)*(d + Vn_qm1[q] - Vn[q])
        # For q == Q: no bid fill, depth irrelevant -> 0
        q_idx = np.arange(nQ)
        # interior q (q < Q, i.e. index < nQ-1)
        interior = slice(0, nQ - 1)
        coeff_bid = (Vn_qm1[interior] - Vn[interior])  # (N_D-1,)
        # objective[d, q] = Vn[q] + p_grid[d] * (depth_grid[d] + coeff_bid[q])
        # To save memory, process in q-batches
        batch = 64
        for s in range(0, nQ - 1, batch):
            e = min(s + batch, nQ - 1)
            c = coeff_bid[s:e]  # (B,)
            # obj[d, b] = p[d]*(depth[d] + c[b])  (Vn constant per q, doesn't affect argmax)
            obj = p_grid[:, None] * (depth_grid[:, None] + c[None, :])
            idx = np.argmax(obj, axis=0)
            # parabolic refinement
            ref = np.empty(e - s, dtype=np.float64)
            for j in range(e - s):
                ref[j] = _parabolic_refine(obj[:, j], depth_grid, idx[j])
            bid_row[s:e] = ref

        # --- Optimize ask depth for each q ---
        # For q > -Q: maximize over d: d*p(d) + p(d)*Vn_qp1[q] + (1-p(d))*Vn[q]
        # For q == -Q: no ask fill -> 0
        coeff_ask = (Vn_qp1[1:] - Vn[1:])  # q in 0..Q, i.e. index 1..nQ-1
        for s in range(1, nQ, batch):
            e = min(s + batch, nQ)
            c = coeff_ask[s - 1:e - 1]
            obj = p_grid[:, None] * (depth_grid[:, None] + c[None, :])
            idx = np.argmax(obj, axis=0)
            ref = np.empty(e - s, dtype=np.float64)
            for j in range(e - s):
                ref[j] = _parabolic_refine(obj[:, j], depth_grid, idx[j])
            ask_row[s:e] = ref

        bid_table[n] = bid_row
        ask_table[n] = ask_row

        # --- Update V[n, q] ---
        # V[n,q] = penalty[q] + bid_contrib + ask_contrib
        # bid_contrib for q<Q: max_d [d p + p Vn_qm1 + (1-p) Vn] = Vn[q] + max_d[p*(d + Vn_qm1 - Vn)]
        # For q==Q: 0
        # ask_contrib for q>-Q: similar; for q==-Q: 0
        Vn_q = Vn  # (nQ,)
        bid_contrib = np.zeros(nQ, dtype=np.float64)
        ask_contrib = np.zeros(nQ, dtype=np.float64)

        # Recompute max values (we have the refined depths; recompute objective at refined depth)
        # For q < Q:
        p_b = 1.0 - np.exp(-A * np.exp(-k * bid_row[:nQ - 1]) * dt)
        bid_contrib[:nQ - 1] = (p_b * (bid_row[:nQ - 1] + Vn_qm1[:nQ - 1])
                                + (1.0 - p_b) * Vn_q[:nQ - 1])
        # For q > -Q:
        p_a = 1.0 - np.exp(-A * np.exp(-k * ask_row[1:]) * dt)
        ask_contrib[1:] = (p_a * (ask_row[1:] + Vn_qp1[1:])
                           + (1.0 - p_a) * Vn_q[1:])

        V[n] = penalty + bid_contrib + ask_contrib

    return bid_table, ask_table


def _get_tables(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _HJB_CACHE:
        return _HJB_CACHE[key]
    deadline = _DEADLINE
    tables = _solve_hjb(params["sigma"], params["A"], params["k"],
                        params["gamma"], params["T"], params["steps"],
                        params["Q"], deadline)
    _HJB_CACHE[key] = tables
    return tables


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (N,) from precomputed HJB."""
    n = int(round(t / (params["T"] / params["steps"])))
    steps = params["steps"]
    Q = params["Q"]
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    bid_table, ask_table = _get_tables(params)

    # Map inventory q in [-Q, Q] to index q + Q, clipped
    q = np.asarray(inventory, dtype=np.int64)
    idx = np.clip(q + Q, 0, 2 * Q)

    bid = bid_table[n, idx].astype(np.float64)
    ask = ask_table[n, idx].astype(np.float64)

    # Enforce boundary: no bid fill at q = Q -> bid depth 0 (irrelevant but safe)
    # no ask fill at q = -Q -> ask depth 0
    bid = np.where(q >= Q, 0.0, bid)
    ask = np.where(q <= -Q, 0.0, ask)

    # Clamp to valid range: must be >= -3/k (task says below -3/k invalid), and finite
    min_allowed = -3.0 / params["k"]
    bid = np.maximum(bid, min_allowed)
    ask = np.maximum(ask, min_allowed)
    bid = np.where(np.isfinite(bid), bid, min_allowed)
    ask = np.where(np.isfinite(ask), ask, min_allowed)

    return bid, ask
# EVOLVE-BLOCK-END
