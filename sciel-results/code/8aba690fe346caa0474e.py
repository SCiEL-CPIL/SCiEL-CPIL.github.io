"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

# Deadline: 140s (safety margin before 150s kill)
_DEADLINE = 140.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _score_points(pts):
    """Compute min triangle area / convex hull area. Returns 0.0 on failure."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Ensure points are 2D
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if pts.shape[0] != n:
        return 0.0

    # Convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume == area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # All triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
                            - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area

    return min_area / hull_area


def _neg_score_params(params):
    """Negative score for optimizer (minimizes negative = maximizes score)."""
    pts = params.reshape(-1, 2)
    s = _score_points(pts)
    if s <= 0:
        return 1.0  # large positive = bad
    return -s


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0

    best_score = 0.0
    best_pts = None

    # Initial random points
    rng = np.random.RandomState(42)
    init_pts = rng.rand(N, 2)
    init_score = _score_points(init_pts)
    best_score = init_score
    best_pts = init_pts.copy()

    # Differential evolution for global search
    # Bounds: unit square [0,1]x[0,1]
    bounds = [(0.0, 1.0)] * (2 * N)

    # Check time before DE
    if _remaining() < 30.0:
        return best_pts

    try:
        de_result = differential_evolution(
            _neg_score_params,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            init='sobol',
            polish=False,  # We'll do our own polish
            callback=lambda x, convergence: (
                False if _remaining() < 15.0 else None
            )
        )
        de_pts = de_result.x.reshape(-1, 2)
        de_score = _score_points(de_pts)
        if de_score > best_score:
            best_score = de_score
            best_pts = de_pts.copy()
    except Exception:
        pass

    # Nelder-Mead local refinement
    if _remaining() < 10.0:
        return best_pts

    try:
        nm_result = minimize(
            _neg_score_params,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        nm_pts = nm_result.x.reshape(-1, 2)
        nm_score = _score_points(nm_pts)
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts.copy()
    except Exception:
        pass

    # Final high-precision polish
    if _remaining() < 5.0:
        return best_pts

    try:
        final_result = minimize(
            _neg_score_params,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        final_pts = final_result.x.reshape(-1, 2)
        final_score = _score_points(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
    except Exception:
        pass

    return best_pts
# EVOLVE-BLOCK-END
