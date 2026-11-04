"""Combined program: multi-init L-BFGS-B + Riemannian Armijo line search
+ Gram-matrix (eigenvalue-repair) auxiliary escape, for 150-point log-energy."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
_DEADLINE = 145.0  # explicit wall-clock guard (150s kill, safety margin)


def _log_energy(points):
    """log(1/d) energy via math.fsum (chordal distances)."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(np.maximum(dists_sq, 1e-300))
    iu = np.triu_indices(n, k=1)
    return -math.fsum(np.log(dists[iu]).tolist())


def _fast_energy(points):
    """Fast numpy energy (for optimizers)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    return -np.sum(0.5 * np.log(dists_sq[iu]))


def _grad(points):
    """Tangent-space gradient of E = -sum_{i<j} log d_ij, shape (N,3)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    inv = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv[mask] = 1.0 / dists_sq[mask]
    sum_w = np.sum(inv, axis=1)
    wsum = inv @ points
    grads = wsum - points * sum_w[:, np.newaxis]
    dot = np.sum(grads * points, axis=1, keepdims=True)
    return grads - dot * points


def _project(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _geodesic_dome_init(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    def subdiv(v0, v1, v2, s):
        if s == 1:
            return [(v0, v1, v2)]
        m01 = (v0 + v1) / 2.0; m01 = m01 / np.linalg.norm(m01)
        m12 = (v1 + v2) / 2.0; m12 = m12 / np.linalg.norm(m12)
        m20 = (v2 + v0) / 2.0; m20 = m20 / np.linalg.norm(m20)
        return (subdiv(v0, m01, m20, s - 1) + subdiv(m01, v1, m12, s - 1) +
                subdiv(m20, m12, v2, s - 1) + subdiv(m01, m12, m20, s - 1))

    allv = set()
    for f in faces:
        for sf in subdiv(verts[f[0]], verts[f[1]], verts[f[2]], 2):
            for v in sf:
                allv.add(tuple(np.round(v, 12)))
    res = np.array(list(allv))
    res = res / np.linalg.norm(res, axis=1, keepdims=True)
    rng = np.random.RandomState(42)
    if len(res) > n:
        res = res[rng.choice(len(res), size=n, replace=False)]
    elif len(res) < n:
        res = np.vstack([res, _random_sphere(n - len(res), rng)])
    return res


def _armijo(pts, grad, time_budget, max_iter=800, c=1e-4, rho=0.5,
            alpha0=0.05, alpha_min=1e-12):
    """Riemannian gradient descent with Armijo backtracking line search.
    Adaptive step, tangent direction, re-normalize to sphere. Robust to
    divergence; returns improved points (only accepted if energy drops)."""
    t_end = time.time() + time_budget
    E = _fast_energy(pts)
    best = pts.copy()
    best_E = E
    alpha = alpha0
    for _ in range(max_iter):
        if time.time() > t_end:
            break
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-11:
            break
        # adaptive step from curvature scale
        alpha = max(alpha_min, min(0.5, 0.02 / (gnorm + 1e-8)))
        d = -grad
        d = d - np.sum(d * pts, axis=1, keepdims=True) * pts  # ensure tangent
        dn = np.linalg.norm(d)
        if dn < 1e-14:
            break
        d = d / dn
        # Armijo backtracking
        step = alpha
        for _b in range(30):
            cand = pts + step * d
            cand = _project(cand)
            Ec = _fast_energy(cand)
            if Ec <= E - c * step * (dn * gnorm) or Ec < E:
                break
            step *= rho
        else:
            break
        pts = cand
        E = Ec
        if E < best_E:
            best_E = E
            best = pts.copy()
        grad = _grad(pts)
    return best, best_E


def _gram_repair(pts, eps=1e-6, time_budget=2.0):
    """Auxiliary PSD/Gram-matrix escape: build G=X X^T, clip eigenvalues >= eps,
    re-embed via eigendecomposition, then re-polish. Can lift rank-deficient /
    co-located clusters out of shallow local minima."""
    t_end = time.time() + time_budget
    G = pts @ pts.T
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, eps)
    G2 = (V * w) @ V.T
    w2, V2 = np.linalg.eigh(G2)
    w2 = np.maximum(w2, 0.0)
    # rank-3 embedding: keep top 3 eigenvectors
    idx = np.argsort(w2)[::-1][:3]
    X = V2[:, idx] * np.sqrt(np.maximum(w2[idx], 0.0))
    X = _project(X)
    if time.time() < t_end - 0.2:
        X, _ = _armijo(X, _grad(X), max(0.2, t_end - time.time()), max_iter=150)
    return X


def _lbfgs(pts, time_budget, maxiter=3000, ftol=1e-14, gtol=1e-10):
    """L-BFGS-B with internal sphere projection."""
    def fgrad(x):
        p = _project(x.reshape(N, 3))
        e = _fast_energy(p)
        g = _grad(p)
        return e, g.flatten()
    try:
        res = minimize(fgrad, pts.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 20,
                                'ftol': ftol, 'gtol': gtol, 'maxls': 60})
        p = _project(res.x.reshape(N, 3))
        # Armijo polish to tighten the local minimum
        p, _ = _armijo(p, _grad(p), max(0.2, time_budget - (time.time() - (time.time()))),
                       max_iter=200, alpha0=0.03)
        return p
    except Exception:
        return pts


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = np.random.RandomState(7)

    inits = [
        _geodesic_dome_init(N),
        _fibonacci_sphere(N),
        _random_sphere(N, rng),
        _random_sphere(N, rng),
        _random_sphere(N, rng),
        _random_sphere(N, rng),
    ]

    best = inits[0].copy()
    best_E = _log_energy(best)

    # Phase 1: L-BFGS-B from multiple initializations
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        cand = _lbfgs(init, 25.0, maxiter=3000)
        e = _log_energy(cand)
        if e < best_E:
            best_E, best = e, cand.copy()

    # Phase 2: basin hopping with Armijo re-optimize + occasional Gram escape
    T = 0.12
    for it in range(60):
        if time.time() > deadline - 25.0:
            break
        if T < 1e-6:
            break
        # perturb a random subset
        k = rng.randint(3, 10)
        idx = rng.choice(N, k, replace=False)
        cand = best.copy()
        for i in idx:
            r = rng.randn(3)
            tan = r - np.dot(r, cand[i]) * cand[i]
            tn = np.linalg.norm(tan)
            if tn > 1e-15:
                cand[i] = cand[i] + T * tan / tn
        cand = _project(cand)
        cand, _ = _armijo(cand, _grad(cand), 6.0, max_iter=300)
        e = _log_energy(cand)
        if e < best_E or (T > 0 and rng.rand() < math.exp(-(e - best_E) / T)):
            if e < best_E:
                best_E, best = e, cand.copy()
            else:
                best = cand.copy()
        T *= 0.92

    # Phase 3: Gram-matrix escape attempts from the current best
    for _g in range(6):
        if time.time() > deadline - 12.0:
            break
        g = _gram_repair(best, eps=1e-5, time_budget=4.0)
        e = _log_energy(g)
        if e < best_E:
            best_E, best = e, g.copy()

    # Phase 4: high-precision polish (tighter tolerances)
    if time.time() < deadline - 6.0:
        best = _lbfgs(best, 5.0, maxiter=6000, ftol=1e-16, gtol=1e-13)
        best, _ = _armijo(best, _grad(best), 4.0, max_iter=600, alpha0=0.02)
        best_E = _log_energy(best)
    if time.time() < deadline - 2.0:
        best, _ = _armijo(best, _grad(best), 2.0, max_iter=1000,
                          alpha0=0.01, alpha_min=1e-14)
        best_E = _log_energy(best)

    best = _project(best)
    if not np.all(np.isfinite(best)):
        return _fibonacci_sphere(N)
    # validity: no pair closer than 1e-9
    diff = best[:, np.newaxis, :] - best[np.newaxis, :, :]
    dsq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dsq[iu]) < (1e-9) ** 2:
        return _fibonacci_sphere(N)
    return best
