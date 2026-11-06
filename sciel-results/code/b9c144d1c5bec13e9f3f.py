"""QAOA angle design: analytic expected cut + exhaustive optimum + grid/Nelder-Mead search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_TIME_BUDGET = 120.0


def _laplacian_eigs(n, edges):
    if n == 1:
        return np.zeros(1)
    L = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        u = int(u); v = int(v)
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            continue
        L[u, v] -= 1.0
        L[v, u] -= 1.0
        L[u, u] += 1.0
        L[v, v] += 1.0
    w = np.linalg.eigvalsh(L)
    return w


def _expected_cut(n, w, g1, g2, b1, b2):
    s1 = np.sin(2.0 * g1)
    s2 = np.sin(2.0 * g2)
    c1 = np.cos(2.0 * g1)
    c2 = np.cos(2.0 * g2)
    t1 = np.tan(b1)
    t2 = np.tan(b2)
    A = c1 * c2 * (1.0 + t1 * t2)
    B = c1 * t2 + c2 * t1
    C = s1 * s2
    D = s1 * t2 + s2 * t1
    S = np.dot(w, w)
    return 0.5 * (A * S + B * S + C * S + D * S)


def _optimum(n, edges):
    if n == 1:
        return 0.0
    e = [(int(u), int(v)) for (u, v) in edges
         if 0 <= int(u) < n and 0 <= int(v) < n and int(u) != int(v)]
    if not e:
        return 0.0
    e = np.array(e, dtype=np.int64)
    m = 1 << (n - 1)
    best = 0.0
    for start in range(0, m, 16384):
        stop = min(start + 16384, m)
        idx = np.arange(start, stop, dtype=np.int64)[:, None]
        bits = ((idx >> np.arange(1, n)) & 1).astype(np.int64)
        bits = np.concatenate([np.zeros((stop - start, 1), dtype=np.int64), bits], axis=1)
        d = bits[:, e[:, 0]] - bits[:, e[:, 1]]
        cut = (d != 0).sum(axis=1)
        b = int(cut.max())
        if b > best:
            best = b
    return float(best)


def angles(n, edges, p):
    t0 = time.time()
    n = int(n)
    w = _laplacian_eigs(n, edges)
    opt = _optimum(n, edges)
    if opt <= 0.0:
        return np.zeros((p, 2), dtype=float)

    def ratio(x):
        return _expected_cut(n, w, x[0], x[1], x[2], x[3]) / opt

    # Global coarse grid over a graph-scaled parameter space.
    g_lo, g_hi = 0.05, 1.2
    b_lo, b_hi = 0.05, 1.2
    n_g = 24
    n_b = 24
    g_grid = np.linspace(g_lo, g_hi, n_g)
    b_grid = np.linspace(b_lo, b_hi, n_b)

    best_x = np.array([0.35, 0.35, 0.25, 0.25], dtype=float)
    best_v = ratio(best_x)

    for i in range(n_g):
        for j in range(n_b):
            x = np.array([g_grid[i], g_grid[j], b_grid[i], b_grid[j]], dtype=float)
            v = ratio(x)
            if v > best_v:
                best_v = v
                best_x = x
            if (time.time() - t0) > _TIME_BUDGET - 8.0:
                return best_x.reshape(p, 2)

    # High-precision Nelder-Mead refinement from the best grid point.
    try:
        from scipy.optimize import minimize
        res = minimize(ratio, best_x.copy(), method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-12,
                                'adaptive': True})
        if res.x is not None and np.all(np.isfinite(res.x)):
            xv = ratio(res.x)
            if xv > best_v:
                best_x = res.x.copy()
                best_v = xv
    except Exception:
        pass

    if (time.time() - t0) > _TIME_BUDGET - 5.0:
        return best_x.reshape(p, 2)

    # Final polish with tighter tolerances.
    try:
        from scipy.optimize import minimize
        res = minimize(ratio, best_x.copy(), method='Nelder-Mead',
                       options={'maxiter': 1200, 'xatol': 1e-11, 'fatol': 1e-14,
                                'adaptive': True})
        if res.x is not None and np.all(np.isfinite(res.x)):
            xv = ratio(res.x)
            if xv > best_v:
                best_x = res.x.copy()
                best_v = xv
    except Exception:
        pass

    out = np.zeros((p, 2), dtype=float)
    out[0] = [best_x[0], best_x[2]]
    out[1] = [best_x[1], best_x[3]]
    return out
# EVOLVE-BLOCK-END
