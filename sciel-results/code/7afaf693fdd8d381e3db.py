"""Minimise logarithmic energy of 150 points on the unit sphere."""
import math
import time
import numpy as np

N = 150
# Hard evaluator wall-clock limit; keep a safety margin before it.
DEADLINE = 138.0  # seconds (evaluator kills at 150s)
_START = time.perf_counter()
_EPS = 1e-9


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum on pair terms."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    terms = []
    for i in range(n):
        diff = pts[i + 1:] - pts[i]
        d = np.linalg.norm(diff, axis=1)
        d = np.maximum(d, _EPS)
        terms.extend(math.log(1.0 / v) for v in d)
    return math.fsum(terms)


def _energy_grad(pts):
    """Vectorised energy and gradient (Euclidean) of log energy.

    Returns (E, g) where g[i] = dE/dx_i.  A small clamp on distances keeps the
    gradient finite for near-coincident points.
    """
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]            # (n, n, 3)
    d = np.linalg.norm(diff, axis=2)                    # (n, n)
    np.fill_diagonal(d, 1.0)
    d = np.maximum(d, 1e-6)                             # clamp to avoid blow-up
    # log(1/d) = -log(d); dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|
    g = (diff / d[:, :, None]).sum(axis=1)              # (n, 3)
    E = -0.5 * np.sum(np.log(d))                        # off-diagonal double counted
    return E, g


def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _cg_minimize(pts, max_iter, time_budget):
    """Conjugate gradient on the sphere via tangent projection.

    Returns (best_pts, best_E).
    """
    pts = np.array(pts, dtype=np.float64, copy=True)
    E, g = _energy_grad(pts)
    best_pts = pts.copy()
    best_E = E
    p = -g
    it = 0
    while it < max_iter:
        if time.perf_counter() - _START > time_budget:
            break
        # Project p onto tangent plane at each point.
        p = p - (p * pts).sum(axis=1, keepdims=True) * pts
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-14:
            break
        p = p / p_norm
        # Backtracking line search.
        step = 1e-2
        for _ in range(30):
            cand = pts + step * p
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cand_E, cand_g = _energy_grad(cand)
            if math.isfinite(cand_E) and cand_E < E:
                # Armijo curvature check: gradient descent direction.
                if np.sum(cand_g * p) < 0:
                    break
            step *= 0.5
        else:
            break
        if not math.isfinite(cand_E):
            break
        # Move.
        pts = cand
        g_new = cand_g
        E = cand_E
        if E < best_E:
            best_E = E
            best_pts = pts.copy()
        # Fletcher-Reeves update of the conjugate direction.
        beta = float(np.sum(g_new * g_new) / max(np.sum(g * g), 1e-14))
        p = -g_new + beta * p
        p = p - (p * pts).sum(axis=1, keepdims=True) * pts  # re-project
        # Periodic re-orthogonalisation to prevent drift.
        if it % 20 == 0:
            p = -g_new
        g = g_new
        it += 1
    return best_pts, best_E


def construct():
    time_budget = DEADLINE - (time.perf_counter() - _START) - 2.0  # 2s safety margin
    if time_budget < 1.0:
        return _fibonacci_sphere()

    rng = np.random.default_rng(2024)
    starts = [_fibonacci_sphere()]
    for _ in range(6):
        starts.append(_random_sphere(rng))

    best_pts = starts[0]
    best_E = _log_energy(starts[0])

    # Roughly equal time budget per start.
    per_start = time_budget / len(starts)
    for s in starts:
        if time.perf_counter() - _START > DEADLINE - 3.0:
            break
        t0 = time.perf_counter()
        max_iter = 4000
        cand_pts, cand_E = _cg_minimize(s, max_iter, t0 + per_start)
        if math.isfinite(cand_E) and cand_E < best_E:
            best_E = cand_E
            best_pts = cand_pts

    # Final normalisation / validation.
    best_pts = np.asarray(best_pts, dtype=np.float64)
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / np.maximum(norms, 1e-12)
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere()
    return best_pts
