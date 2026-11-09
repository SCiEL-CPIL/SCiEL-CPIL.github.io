# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_START_TIME = time.time()
_TIME_LIMIT = 600.0  # seconds; leave safety margin
_N = 20


def _vertex_score(d):
    """Compute the per-vertex score given sorted distances d (length 19)."""
    if d[0] < 0.05:
        return 0.0
    # find first index k minimizing d_{k+1} - d_k
    diffs = d[1:] - d[:-1]
    k = int(np.argmin(diffs))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = np.array([0, 1, 2, 3])
    elif k == 17:
        idx = np.array([15, 16, 17, 18])
    else:
        idx = np.array([k - 1, k, k + 1, k + 2])
    four = d[idx]
    raw = np.mean(np.maximum(four / D, D / four))
    return min(1.0, 1.0 / raw)


def _score_polygon(pts):
    """
    Score a (20,2) array of vertices. Returns (score, is_valid).
    Normalizes: subtract centroid, divide by mean distance to centroid.
    Checks convexity. Computes min vertex score.
    """
    if pts.shape != (20, 2):
        return 0.0, False
    if not np.all(np.isfinite(pts)):
        return 0.0, False

    # Normalize
    centroid = pts.mean(axis=0)
    p = pts - centroid
    dists_to_centroid = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0, False
    p = p / mean_dist

    # Check no repeated points
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return 0.0, False

    # Check strictly convex: all cross products same sign, nonzero
    cross_signs = []
    for i in range(20):
        a = p[i]
        b = p[(i + 1) % 20]
        c = p[(i + 2) % 20]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-12:
            return 0.0, False
        cross_signs.append(np.sign(cross))
    if len(set(cross_signs)) != 1:
        return 0.0, False

    # Compute per-vertex scores
    scores = []
    for i in range(20):
        d = np.sort(np.linalg.norm(p - p[i], axis=1))
        d = d[1:]  # exclude self
        scores.append(_vertex_score(d))

    min_score = min(scores)
    return min_score, True


def _build_polygon(params):
    """
    Build polygon from Fourier angular deviation parameters.
    params: array of 6 values: [a1, b1, a2, b2, a3, b3]
    Angular deviation: delta_i = a1*cos(2*pi*i/20) + b1*sin(2*pi*i/20)
                       + a2*cos(4*pi*i/20) + b2*sin(4*pi*i/20)
                       + a3*cos(6*pi*i/20) + b3*sin(6*pi*i/20)
    Base angle: 2*pi*i/20
    Radius: 1.0 (scale-invariant)
    """
    i = np.arange(20)
    base_angles = 2.0 * np.pi * i / 20.0
    a1, b1, a2, b2, a3, b3 = params
    delta = (a1 * np.cos(2 * np.pi * i / 20.0) + b1 * np.sin(2 * np.pi * i / 20.0)
             + a2 * np.cos(4 * np.pi * i / 20.0) + b2 * np.sin(4 * np.pi * i / 20.0)
             + a3 * np.cos(6 * np.pi * i / 20.0) + b3 * np.sin(6 * np.pi * i / 20.0))
    angles = base_angles + delta
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)


def _objective(params):
    """Minimize negative score (with validity penalty)."""
    pts = _build_polygon(params)
    score, valid = _score_polygon(pts)
    if not valid:
        return 1.0  # large penalty
    return -score


def _regular_polygon():
    """Fallback: regular 20-gon."""
    i = np.arange(20)
    angles = 2.0 * np.pi * i / 20.0
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum vertex equidistance score.
    Uses Fourier angular deviation parameterization with differential evolution.
    """
    # Bounds for Fourier coefficients: keep deviations small to maintain convexity
    bounds = [(-0.15, 0.15)] * 6

    # Initial best: regular polygon
    best_pts = _regular_polygon()
    best_score, _ = _score_polygon(best_pts)

    # Try differential evolution if time permits
    elapsed = time.time() - _START_TIME
    if elapsed < _TIME_LIMIT - 60:
        try:
            # Use a subset of the time budget for DE
            time_budget = min(_TIME_LIMIT - 60 - elapsed, 400)
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                init='sobol',
                updating='immediate',
                maxfev=int(20000),
            )
            if result.success or result.fun < 0:
                candidate_pts = _build_polygon(result.x)
                cand_score, cand_valid = _score_polygon(candidate_pts)
                if cand_valid and cand_score > best_score:
                    best_pts = candidate_pts
                    best_score = cand_score
        except Exception:
            pass

    # Check time
    elapsed = time.time() - _START_TIME
    if elapsed < _TIME_LIMIT - 30:
        # Try a local refinement with Nelder-Mead on the best found
        from scipy.optimize import minimize
        try:
            # Get current params if we have them, else use zero (regular)
            if hasattr(equidistant_points_polygon20, '_last_params'):
                x0 = equidistant_points_polygon20._last_params
            else:
                x0 = np.zeros(6)

            # Wrap objective to track best
            def _obj_local(p):
                pts = _build_polygon(p)
                s, v = _score_polygon(pts)
                if not v:
                    return 1.0
                return -s

            res = minimize(_obj_local, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < 0:
                candidate_pts = _build_polygon(res.x)
                cand_score, cand_valid = _score_polygon(candidate_pts)
                if cand_valid and cand_score > best_score:
                    best_pts = candidate_pts
                    best_score = cand_score
        except Exception:
            pass

    # Final time check
    if time.time() - _START_TIME > _TIME_LIMIT - 10:
        return best_pts

    return best_pts


# EVOLVE-BLOCK-END
