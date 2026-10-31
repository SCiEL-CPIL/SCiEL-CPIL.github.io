import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x_flat, beta=60.0):
    """Smooth surrogate for -min(triangle_area)/hull_area.

    Uses a shifted soft-min over the area ratios so the landscape is
    differentiable-friendly for population-based and gradient-free methods.
    """
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10

    # Keep the configuration within a sane bound (scale-invariant objective,
    # but avoid degenerate blow-ups).
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm > 3.0:
        return 1e10 + (max_norm - 3.0) ** 2

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10

    areas = _triangle_areas_vectorized(points)
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10

    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10

    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _hard_objective(x_flat):
    """Hard objective: -min_triangle_area / hull_area."""
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _regular_polygon(n, radius=1.0, offset=0.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _rings(counts, radii, offsets):
    """Build a configuration from concentric regular rings."""
    parts = []
    for c, r, a in zip(counts, radii, offsets):
        if c <= 0:
            continue
        angles = np.linspace(a, 2 * np.pi + a, c, endpoint=False)
        parts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    return np.vstack(parts)


def _generate_templates():
    """Generate diverse structured initial configurations (the structural idea)."""
    t = []

    # 1. Regular 13-gon (a few angular offsets)
    for off in [0.0, np.pi / 13.0]:
        t.append(_regular_polygon(N, 1.0, off))

    # 2. Concentric rings: 1 center + 6 inner + 6 outer
    t.append(_rings([1, 6, 6], [0.0, 0.5, 1.0], [0.0, 0.0, np.pi / 6.0]))
    # 3. 1 center + 12 outer
    t.append(_rings([1, 12], [0.0, 1.0], [0.0, 0.0]))
    # 4. 1 center + 4 inner + 8 outer
    t.append(_rings([1, 4, 8], [0.0, 0.45, 1.0], [0.0, np.pi / 4.0, 0.0]))
    # 5. 7 outer + 6 inner
    t.append(_rings([7, 6], [1.0, 0.5], [0.0, np.pi / 6.0]))
    # 6. 6 outer + 7 inner
    t.append(_rings([6, 7], [1.0, 0.55], [0.0, np.pi / 7.0]))
    # 7. 5 outer + 4 middle + 4 inner
    t.append(_rings([5, 4, 4], [1.0, 0.6, 0.25], [0.0, np.pi / 5.0, np.pi / 4.0]))
    # 8. 8 outer + 5 inner
    t.append(_rings([8, 5], [1.0, 0.5], [0.0, np.pi / 5.0]))
    # 9. 9 outer + 4 inner
    t.append(_rings([9, 4], [1.0, 0.45], [0.0, np.pi / 4.0]))
    # 10. 10 outer + 3 inner
    t.append(_rings([10, 3], [1.0, 0.4], [0.0, np.pi / 3.0]))
    # 11. 11 outer + 2 inner
    t.append(_rings([11, 2], [1.0, 0.35], [0.0, 0.0]))
    # 12. 12 outer + 1 inner
    t.append(_rings([12, 1], [1.0, 0.3], [0.0, 0.0]))

    return t


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin against the 150 s limit

    best_points = None
    best_score = 0.0

    def _elapsed():
        return time.time() - start_time

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    def _polish(x0, maxiter, xatol, fatol, beta=None, hard=False):
        """Local Nelder-Mead polish; returns improved points or None."""
        if _elapsed() > deadline - 8:
            return None
        try:
            if hard:
                res = minimize(
                    _hard_objective, x0, method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xatol,
                             'fatol': fatol, 'adaptive': True})
            else:
                res = minimize(
                    _soft_min_objective, x0, args=(beta,), method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xatol,
                             'fatol': fatol, 'adaptive': True})
            return res.x.reshape(N, 2)
        except Exception:
            return None

    # Phase 1: evaluate structured templates
    templates = _generate_templates()
    for g in templates:
        _update_best(g)

    # Phase 2: global search with differential evolution (the new search method).
    # Population-based, so it explores far more of the 26-dim space than a
    # single-trajectory Nelder-Mead and escapes the local minima that trap it.
    bounds = [(-1.5, 1.5)] * (2 * N)
    if _elapsed() < deadline - 40:
        try:
            def _ev_obj(x):
                return _soft_min_objective(x, beta=60.0)

            def _ev_cb(xk, convergence):
                pts = xk.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                return _elapsed() > deadline - 20

            de = differential_evolution(
                _ev_obj, bounds, seed=42, maxiter=40, popsize=18,
                tol=1e-6, mutation=(0.4, 1.0), recombination=0.8,
                init='sobol', polish=False, callback=_ev_cb,
                updating='deferred', x0=None, disp=False)
            if de.x is not None:
                _update_best(de.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 3: local polish from the best point and a few strong templates.
    if best_points is not None:
        for beta in [40.0, 15.0]:
            p = _polish(best_points.flatten(), 800, 1e-8, 1e-11, beta=beta)
            if p is not None:
                _update_best(p)
        p = _polish(best_points.flatten(), 2500, 1e-10, 1e-12, hard=True)
        if p is not None:
            _update_best(p)

    # Polish a couple of the best templates independently (multi-seed escape).
    for g in templates[:4]:
        if _elapsed() > deadline - 10:
            break
        p = _polish(g.flatten(), 600, 1e-8, 1e-10, beta=30.0)
        if p is not None:
            _update_best(p)

    # Phase 4: random restarts if time remains.
    rng = np.random.default_rng(7)
    while _elapsed() < deadline - 15:
        if best_points is not None:
            pert = best_points + rng.normal(0, 0.12, (N, 2))
            _update_best(pert)
            p = _polish(pert.flatten(), 400, 1e-8, 1e-10, beta=25.0)
            if p is not None:
                _update_best(p)
        else:
            break

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
