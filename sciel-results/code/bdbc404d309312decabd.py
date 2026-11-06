"""QAOA p=2 angle design: exact expectation with dense grid + multi-start Nelder-Mead."""
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 135.0


def _remaining():
    return _DEADLINE - time.time()


def _build_graph(n, edges):
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        if u == v:
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(n, edges, g1, b1, g2, b2, cache):
    """Vectorized expected MaxCut value for p=2 QAOA."""
    m = len(edges)
    if m == 0:
        return 0.0

    key = (n, tuple(edges))
    if key not in cache:
        c = 1 << n
        u = np.empty(m, dtype=np.int64)
        v = np.empty(m, dtype=np.int64)
        for k, (a, b) in enumerate(edges):
            u[k] = a
            v[k] = b
        bits = np.arange(c, dtype=np.int64)
        zu = np.where((bits >> u) & 1, -1.0, 1.0)
        zv = np.where((bits >> v) & 1, -1.0, 1.0)
        zz = zu * zv
        cache[key] = (c, zz)
    c, zz = cache[key]

    cos1, sin1 = np.cos(b1), np.sin(b1)
    cos2, sin2 = np.cos(b2), np.sin(b2)

    phase1 = np.exp(0.5j * g1 * zz)
    p = (cos1 + 1j * sin1 * zz) * phase1
    phase2 = np.exp(0.5j * g2 * zz)
    p = p * phase2

    prob = np.abs(p) ** 2
    corr = np.sum(prob * zz, axis=1)
    return float(np.sum(0.5 * (1.0 - corr)))


def _objective(params, n, edges, cache):
    return -_expected_cut(n, edges, params[0], params[1], params[2], params[3], cache)


def angles(n, edges, p):
    adj = _build_graph(n, edges)
    m = len(edges)
    cache = {}

    # Compute graph properties
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        w = np.linalg.eigvalsh(L)
        lmax = float(w[-1]) if len(w) else 0.0
    except Exception:
        lmax = 0.0

    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    # Graph-aware parameter ranges
    if lmax > 1e-8:
        gamma_max = min(2.5, 0.8 * lmax)
    else:
        gamma_max = 0.5
    beta_max = 1.5

    # Dense grid search for global coverage
    n_grid = 12
    gamma_vals = np.linspace(0.01, gamma_max, n_grid)
    beta_vals = np.linspace(0.01, beta_max, n_grid)

    best_val = -np.inf
    best_x = np.array([0.2, 0.2, 0.2, 0.2])
    top_points = []

    # Phase 1: Coarse grid
    for g1 in gamma_vals:
        if _remaining() < 30.0:
            break
        for b1 in beta_vals:
            if _remaining() < 30.0:
                break
            for g2 in gamma_vals:
                if _remaining() < 30.0:
                    break
                for b2 in beta_vals:
                    try:
                        val = _expected_cut(n, edges, g1, b1, g2, b2, cache)
                        if val > best_val:
                            best_val = val
                            best_x = np.array([g1, b1, g2, b2])
                        top_points.append((val, [g1, b1, g2, b2]))
                    except Exception:
                        continue

    # Keep top 15 points from grid
    top_points.sort(key=lambda x: x[0], reverse=True)
    top_k = top_points[:15]

    # Phase 2: Refine top candidates with Nelder-Mead
    for val, point in top_k:
        if _remaining() < 20.0:
            break
        try:
            res = minimize(
                _objective, point, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-start from diverse random points
    rng = np.random.default_rng(713)
    for _ in range(20):
        if _remaining() < 15.0:
            break
        g1 = rng.uniform(0.02, gamma_max)
        b1 = rng.uniform(0.02, beta_max)
        g2 = rng.uniform(0.02, gamma_max)
        b2 = rng.uniform(0.02, beta_max)
        point = [g1, b1, g2, b2]
        try:
            res = minimize(
                _objective, point, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision refinement
    if _remaining() > 10.0:
        try:
            res = minimize(
                _objective, best_x, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 5: Perturbation-based exploration
    if _remaining() > 10.0:
        rng2 = np.random.default_rng(123)
        for _ in range(8):
            if _remaining() < 8.0:
                break
            perturbed = best_x + rng2.normal(0, 0.08, size=4)
            perturbed = np.clip(perturbed, 0.01, max(gamma_max, beta_max))
            try:
                res = minimize(
                    _objective, perturbed, args=(n, edges, cache),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True},
                )
                if res.success and -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    out = np.zeros((p, 2), dtype=float)
    out[0, 0] = best_x[0]
    out[0, 1] = best_x[1]
    if p > 1:
        out[1, 0] = best_x[2]
        out[1, 1] = best_x[3]
    return out
