"""Heilbronn-11: maximize min triangle area. Combines soft-min gradient descent
with differential evolution and local Nelder-Mead polishing."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _min_triangle_area(pts):
    return np.min(_all_triangle_areas(pts))


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _bary_to_cart(u, v):
    """Convert 2D barycentric (u,v) to cartesian. w=1-u-v."""
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert cartesian to 2D barycentric (u,v)."""
    v = pts[:, 1] / S3H
    u = pts[:, 0] - 0.5 * v
    return np.column_stack([u, v])


def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return np.array([u + 0.5 * v, v * S3H])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/3, 0.0],
        [2/3, 0.0],
        [0.5, S3H/3],
        [0.5, 2*S3H/3],
    ], dtype=float)
    _update_best(init)

    # ---- Phase A: Differential Evolution in cartesian space ----
    # Use bounds that keep points roughly inside the triangle
    # The triangle has x in [0,1], y in [0, S3H]
    # We'll use a slightly relaxed box and rely on projection
    bounds_de = [(0.0, 1.0), (0.0, S3H)] * N

    def _de_obj(flat):
        pts = flat.reshape(N, 2)
        pts = _project_to_triangle(pts)
        return -_min_triangle_area(pts)

    if _elapsed_ok(60):
        try:
            de_res = differential_evolution(
                _de_obj,
                bounds_de,
                maxiter=60,
                popsize=12,
                seed=42,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updateinterval=0,
                callback=None,
            )
            de_pts = _project_to_triangle(de_res.x.reshape(N, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Multi-start soft-min gradient optimization ----
    # Build diverse starting points
    starts = []
    if best_pts is not None:
        starts.append(best_pts.copy())
    starts.append(init.copy())
    for _ in range(15):
        starts.append(_project_to_triangle(init + rng.normal(0, 0.2, size=(N, 2))))
    for _ in range(10):
        rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        starts.append(rand_pts)

    for s_idx, base in enumerate(starts):
        if not _elapsed_ok(25):
            break
        pts = _project_to_triangle(base + rng.normal(0, 0.03, size=(N, 2)))
        _update_best(pts)

        # Annealed soft-min optimization
        for beta in [5.0, 15.0, 40.0, 100.0, 300.0, 800.0]:
            if not _elapsed_ok(25):
                break

            def objective(x):
                p = _project_to_triangle(x.reshape(N, 2))
                return -_soft_min_area(p, beta)

            try:
                res = minimize(objective, pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11})
                pts = _project_to_triangle(res.x.reshape(N, 2))
                _update_best(pts)
            except Exception:
                pass

    # ---- Phase C: Nelder-Mead local polishing ----
    if best_pts is not None and _elapsed_ok(15):
        # Multiple NM restarts from perturbed best
        for nm_iter in range(12):
            if not _elapsed_ok(10):
                break
            scale = 0.02 * (0.7 ** (nm_iter // 4))
            cand = _project_to_triangle(best_pts + rng.normal(0, scale, size=(N, 2)))

            def nm_obj(x):
                p = _project_to_triangle(x.reshape(N, 2))
                return -_soft_min_area(p, 1500.0)

            try:
                res = minimize(nm_obj, cand.flatten(), method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13,
                                        'adaptive': True})
                p = _project_to_triangle(res.x.reshape(N, 2))
                _update_best(p)
            except Exception:
                pass

    # ---- Phase D: Final fine polish ----
    if best_pts is not None and _elapsed_ok(8):
        def final_obj(x):
            p = _project_to_triangle(x.reshape(N, 2))
            return -_soft_min_area(p, 3000.0)
        try:
            res = minimize(final_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15,
                                    'adaptive': True})
            p = _project_to_triangle(res.x.reshape(N, 2))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        return _project_to_triangle(pts)

    return _project_to_triangle(best_pts)
