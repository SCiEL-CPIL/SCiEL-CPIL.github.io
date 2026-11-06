"""QAOA p=2 angle design: multi-start Nelder-Mead with graph-aware initialization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 140s out of 150s limit, leaving 10s safety margin


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


def _exhaustive_maxcut(n, edges):
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, edges, cache):
    return -_expected_cut(n, edges, params[0], params[1], params[2], params[3], cache)


def _generate_initial_points(n, edges, adj, density, lmax):
    """Generate diverse initial points for multi-start optimization."""
    points = []
    
    # Graph-aware scaling
    if lmax > 1e-8:
        gamma_max = min(1.5, 0.5 * lmax)
    else:
        gamma_max = 0.5
    
    beta_max = 0.8
    
    # Deterministic grid-based points
    n_g = 8
    n_b = 8
    gamma_vals = np.linspace(0.05, gamma_max, n_g)
    beta_vals = np.linspace(0.05, beta_max, n_b)
    
    # Add structured points
    for g1 in gamma_vals[::2]:
        for b1 in beta_vals[::2]:
            for g2 in gamma_vals[::2]:
                for b2 in beta_vals[::2]:
                    points.append([g1, b1, g2, b2])
    
    # Add some random points for diversity
    rng = np.random.RandomState(42)
    for _ in range(20):
        g1 = rng.uniform(0.05, gamma_max)
        b1 = rng.uniform(0.05, beta_max)
        g2 = rng.uniform(0.05, gamma_max)
        b2 = rng.uniform(0.05, beta_max)
        points.append([g1, b1, g2, b2])
    
    # Add some symmetric points (g1≈g2, b1≈b2)
    for g in gamma_vals[::3]:
        for b in beta_vals[::3]:
            points.append([g, b, g, b])
    
    return points


def angles(n, edges, p):
    adj = _build_graph(n, edges)
    m = len(edges)
    cache = {}
    exact = _exhaustive_maxcut(n, edges)
    
    # Compute graph-dependent scaling factors
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        w = np.linalg.eigvalsh(L)
        lmax = float(w[-1]) if len(w) else 0.0
    except Exception:
        lmax = 0.0
    
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # Generate initial points
    initial_points = _generate_initial_points(n, edges, adj, density, lmax)
    
    best_val = -np.inf
    best_x = np.array([0.2, 0.2, 0.2, 0.2])
    
    # Multi-start Nelder-Mead optimization
    for i, x0 in enumerate(initial_points):
        if _remaining() < 5.0:
            break
        
        try:
            res = minimize(
                _objective, np.array(x0), args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10},
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision refinement from the best point found
    if _remaining() > 10.0:
        try:
            res = minimize(
                _objective, best_x, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Another refinement if time permits
    if _remaining() > 5.0:
        try:
            res = minimize(
                _objective, best_x, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14},
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
# EVOLVE-BLOCK-END
