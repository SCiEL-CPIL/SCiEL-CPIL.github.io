import numpy as np
import time
import scipy.spatial as spat
import scipy.optimize as opt
from itertools import combinations

N = 13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx], dtype=np.int64)
_j_arr = np.array([t[1] for t in _triple_idx], dtype=np.int64)
_k_arr = np.array([t[2] for t in _triple_idx], dtype=np.int64)
_n_triples = len(_triple_idx)


def _triangle_areas(points):
    v1 = points[_j_arr] - points[_i_arr]
    v2 = points[_k_arr] - points[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)


def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def calculate_score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area


def _objective(x):
    """Hard objective for DE: minimize negative score (min triangle area / hull area)."""
    try:
        pts = np.array(x, dtype=np.float64).reshape(N, 2)
        # Remove any degenerate configs
        if np.any(~np.isfinite(pts)):
            return 1e10
        hull_area = _hull_area(pts)
        if hull_area <= 1e-9:
            return 1e10
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return -min_area / hull_area
    except Exception:
        return 1e10


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # Bounds: each coordinate in [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * N)

    # Seed with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    reg_flat = reg_pts.flatten().tolist()

    best_score = calculate_score(reg_pts)
    best_points = reg_pts.copy()

    def _update_best(pts):
        nonlocal best_score, best_points
        sc = calculate_score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()

    # Run differential_evolution on the HARD objective
    # population_size=40, maxiter=25 -> ~1000 evaluations
    # Each eval: hull + 286 triangle areas ~ fast
    try:
        result = opt.differential_evolution(
            _objective,
            bounds,
            x0=reg_flat,
            seed=42,
            maxiter=25,
            popsize=40,
            tol=1e-8,
            polish=False,  # We'll do our own polish; DE's polish uses L-BFGS on non-smooth obj
            init='latinhypercube',
            updating='immediate',
            seed=42,
        )
        if result.fun < 1e9:
            de_pts = result.x.reshape(N, 2)
            _update_best(de_pts)
    except Exception:
        pass

    # If time remains, run a second DE with different seed and tighter bounds
    if time.time() < deadline - 15.0:
        try:
            # Tighter bounds around regular polygon
            bounds_tight = [(-1.2, 1.2)] * (2 * N)
            result2 = opt.differential_evolution(
                _objective,
                bounds_tight,
                x0=reg_flat,
                seed=123,
                maxiter=20,
                popsize=30,
                tol=1e-8,
                polish=False,
                init='latinhypercube',
                updating='immediate',
            )
            if result2.fun < 1e9:
                de2_pts = result2.x.reshape(N, 2)
                _update_best(de2_pts)
        except Exception:
            pass

    # Final polish: Nelder-Mead on the hard objective from best point
    if time.time() < deadline - 5.0:
        try:
            nm_result = opt.minimize(
                _objective,
                best_points.flatten().tolist(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
            )
            if nm_result.fun < 1e9:
                nm_pts = nm_result.x.reshape(N, 2)
                _update_best(nm_pts)
        except Exception:
            pass

    return best_points
