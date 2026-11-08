"""Heilbronn convex regions - 13 points maximizing min triangle area / hull area.

Structural change: global search via differential evolution + Nelder-Mead polish.
"""
import time
import numpy as np
import itertools
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int64)
_I = _TRIPLES[:, 0]
_J = _TRIPLES[:, 1]
_K = _TRIPLES[:, 2]


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _exact_score(pts):
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(pts)
    return float(areas.min() / hull_area)


def _objective(x):
    """Minimize -score (so maximize score)."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1.0
    s = _exact_score(pts)
    return -s


def _regular_polygon(n, rot=0.0, r=1.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    return r * np.column_stack([np.cos(ang), np.sin(ang)])


def _nelder_polish(x0, time_budget):
    """Nelder-Mead local polish with a time budget."""
    start = time.time()
    best_x = x0.copy()
    best_f = _objective(x0)
    try:
        res = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True, 'maxfev': 20000},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            f = _objective(res.x)
            if f < best_f:
                best_f = f
                best_x = res.x.copy()
    except Exception:
        pass
    return best_x, best_f


def _coordinate_polish(x0, time_budget):
    """Derivative-free coordinate descent polish with adaptive steps."""
    start = time.time()
    x = x0.copy()
    base_steps = np.full(2 * N, 1e-3)
    cur = _objective(x)
    for outer in range(200):
        if time.time() - start > time_budget - 2:
            break
        improved = False
        for d in range(2 * N):
            if time.time() - start > time_budget - 2:
                break
            x0d = x[d]
            best_d = x0d
            best_val = cur
            step = base_steps[d]
            for mult in [1.0, 0.5, 0.25, 2.0, 0.1, 4.0]:
                for sign in [-1, 1]:
                    x[d] = x0d + sign * step * mult
                    v = _objective(x)
                    if v < best_val - 1e-14:
                        best_val = v
                        best_d = x[d]
            x[d] = best_d
            if best_d != x0d:
                improved = True
                cur = best_val
                base_steps[d] = min(base_steps[d] * 1.3, 2e-2)
            else:
                base_steps[d] *= 0.6
        if not improved:
            if np.max(base_steps) < 1e-7:
                break
    return x, cur


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill

    best_score = -1.0
    best_points = None

    def consider(pts):
        nonlocal best_score, best_points
        if pts is None or not np.all(np.isfinite(pts)):
            return -1.0
        s = _exact_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # ---- Phase 0: structured initializations ----
    init_pool = []
    init_pool.append(_regular_polygon(N))
    for rot in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        init_pool.append(_regular_polygon(N, rot=rot))
    # Ellipse-based
    for ax, by in [(1.0, 0.85), (1.15, 0.9), (0.9, 1.1), (1.3, 0.7), (0.7, 1.3)]:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        init_pool.append(np.column_stack([ax * np.cos(ang), by * np.sin(ang)]))
    # Deterministic perturbations
    rng = np.random.RandomState(42)
    for trial in range(10):
        base = _regular_polygon(N)
        ang = np.arctan2(base[:, 1], base[:, 0])
        rad = np.linalg.norm(base, axis=1)
        ang_p = np.sort(ang + 0.08 * rng.randn(N))
        rad_p = rad + 0.08 * rng.randn(N)
        rad_p = np.clip(rad_p, 0.3, 2.0)
        init_pool.append(np.column_stack([rad_p * np.cos(ang_p), rad_p * np.sin(ang_p)]))

    for pts in init_pool:
        if time.time() > deadline - 5:
            break
        consider(pts)

    if best_points is None:
        best_points = _regular_polygon(N)
        best_score = _exact_score(best_points)

    # ---- Phase 1: Differential evolution (global search) ----
    # Bounds: points in a square [-3, 3] x [-3, 3]
    bounds = [(-3.0, 3.0)] * (2 * N)
    # Use the best initial point as a seed in the population
    x0_de = best_points.flatten().copy()

    # Time budget for DE: most of the remaining time
    de_time_budget = min(80.0, deadline - time.time() - 30)
    if de_time_budget > 10:
        try:
            # Build initial population: include our best point + random
            rng_de = np.random.RandomState(123)
            pop_size = 30
            init_pop = np.array([x0_de] + [
                rng_de.uniform(-3, 3, size=2 * N) for _ in range(pop_size - 1)
            ])

            de_res = differential_evolution(
                _objective,
                bounds,
                seed=123,
                maxiter=200,
                popsize=pop_size,
                tol=1e-12,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,  # we'll do our own polish
                init='so-and-p',
                x0=init_pop,
                updating='deferred',
            )
            if de_res.x is not None and np.all(np.isfinite(de_res.x)):
                consider(de_res.x.reshape(N, 2))
        except Exception:
            pass

    # ---- Phase 2: Nelder-Mead polish from best ----
    if best_points is not None and time.time() < deadline - 10:
        nm_budget = min(25.0, deadline - time.time() - 8)
        if nm_budget > 5:
            x_pol, f_pol = _nelder_polish(best_points.flatten(), nm_budget)
            consider(x_pol.reshape(N, 2))

    # ---- Phase 3: Coordinate descent fine polish ----
    if best_points is not None and time.time() < deadline - 5:
        cd_budget = min(15.0, deadline - time.time() - 3)
        if cd_budget > 3:
            x_cd, f_cd = _coordinate_polish(best_points.flatten(), cd_budget)
            consider(x_cd.reshape(N, 2))

    # ---- Phase 4: Final Nelder-Mead from the very best ----
    if best_points is not None and time.time() < deadline - 3:
        nm_budget2 = min(10.0, deadline - time.time() - 2)
        if nm_budget2 > 3:
            x_pol2, f_pol2 = _nelder_polish(best_points.flatten(), nm_budget2)
            consider(x_pol2.reshape(N, 2))

    # ---- Final safety ----
    if best_points is None:
        best_points = _regular_polygon(N)

    if not np.all(np.isfinite(best_points)) or _hull_area(best_points) <= 1e-12:
        best_points = _regular_polygon(N)

    return best_points
