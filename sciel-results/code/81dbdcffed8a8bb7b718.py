# EVOLVE-BLOCK-START
import numpy as np

_OPT_TABLES = {}


def quote(t, inventory, mid, params):
    """
    Optimal market making with inventory risk via backward induction on the
    discrete inventory MDP.  The transition kernel depends only on the chosen
    (bid_depth, ask_depth), not on t or mid, so the optimal depth per inventory
    level is time-invariant; we precompute it once per regime and look it up.
    Returns (bid_depths, ask_depths), each shape (len(inventory),); np.inf = no quote.
    """
    sigma = float(params["sigma"])
    A = float(params["A"])
    k = float(params["k"])
    gamma = float(params["gamma"])
    Q = int(params["Q"])
    dt = float(params["T"]) / float(params["steps"])

    cache_key = (round(sigma, 6), round(A, 6), round(k, 6),
                 round(gamma, 6), Q, round(dt, 9))

    table = _OPT_TABLES.get(cache_key)
    if table is None:
        table = _compute_optimal_depths(sigma, A, k, gamma, Q, dt)
        _OPT_TABLES[cache_key] = table

    inv = np.asarray(inventory)
    idx = np.clip(inv, -Q, Q) + Q
    bid_depths = table[0][idx].astype(float)
    ask_depths = table[1][idx].astype(float)
    return bid_depths, ask_depths


def _compute_optimal_depths(sigma, A, k, gamma, Q, dt):
    """
    Value-iterate the finite-horizon inventory MDP.  Returns (bid_table, ask_table)
    indexed by inventory+Q, giving the optimal depth on each side.
    """
    n = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    cost = -(gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt  # (n,)

    # candidate depth grid (0 = no quote)
    grid = np.array([0.0] + list(np.geomspace(1e-3, 4.0, 256)), dtype=float)
    grid = np.unique(np.clip(grid, 0.0, None))
    # cap grid size for speed
    if len(grid) > 96:
        grid = grid[np.linspace(0, len(grid) - 1, 96).astype(int)]
    m = len(grid)
    p = np.clip(1.0 - np.exp(-A * np.exp(-k * grid) * dt), 0.0, 1.0)  # (m,)

    V = np.zeros(n)
    opt = np.zeros((n, 2))

    for _step in range(1000):
        newV = np.zeros(n)
        for i, q in enumerate(q_vals):
            pb_avail = (q < Q)
            pa_avail = (q > -Q)

            V0 = V[i]
            j_m1 = max(0, i - 1)
            j_p1 = min(n - 1, i + 1)
            Vm1 = V[j_m1]
            Vp1 = V[j_p1]
            dV_b = Vm1 - V0
            dV_a = Vp1 - V0
            c = cost[i]

            best = -np.inf
            best_b = 0.0
            best_a = 0.0

            for bi in range(m):
                b = grid[bi]
                pb = p[bi] if pb_avail else 0.0
                const = c + pb * b + V0 + pb * dV_b
                coef_a = (1.0 - pb) * dV_a - pb * dV_b
                if pa_avail:
                    val_a = p * (grid + coef_a)
                    jbest = int(np.argmax(val_a))
                    best_pa_term = val_a[jbest]
                    a_best = grid[jbest]
                else:
                    best_pa_term = 0.0
                    a_best = 0.0
                total = const + best_pa_term
                if total > best:
                    best = total
                    best_b = b
                    best_a = a_best

            newV[i] = best
            opt[i, 0] = best_b
            opt[i, 1] = best_a
        V = newV

    return (opt[:, 0], opt[:, 1])
# EVOLVE-BLOCK-END
