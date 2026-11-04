"""Optimized program for Heilbronn convex regions task."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute all triangle index combinations for vectorized area computation
_TRI_IDX = np.array(list(combinations(range(N), 3)))
_I0 = _TRI_IDX[:, 0]
_I1 = _TRI_IDX[:, 1]
_I2 = _TRI_IDX[:, 2]


def _min_triangle_area(pts):
    """Compute minimum triangle area for N points (vectorized)."""
    p0 = pts[_I0]
    p1 = pts[_I1]
    p2 = pts[_I2]
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
            (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def _jittered_regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0, jitter=0.01, seed=0):
    """Regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    radii = radius + rng.uniform(-jitter, jitter, n)
    angles = angles + rng.uniform(-jitter, jitter, n)
    return np.column_stack([
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles)
    ])


def _optimize_from_start(pts0, maxiter=2000, tol=1e-10):
    """L-BFGS-B optimization from a starting point."""
    res = minimize(
        _neg_score,
        pts0.flatten(),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8}
    )
    return res.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_pts = None
    best_score = -np.inf

    # Strategy 1: Regular 13-gon as initial guess
    pts_reg = _regular_polygon(N, radius=1.0)
    score_reg = _score(pts_reg)
    if score_reg > best_score:
        best_score = score_reg
        best_pts = pts_reg.copy()

    # Strategy 2: Try multiple jittered regular polygons and optimize
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        pts_jit = _jittered_regular_polygon(N, radius=1.0, jitter=0.05, seed=seed)
        score_jit = _score(pts_jit)
        if score_jit > best_score:
            best_score = score_jit
            best_pts = pts_jit.copy()

        # Optimize from this starting point
        if time.time() > deadline - 15:
            break
        try:
            opt_pts = _optimize_from_start(pts_jit, maxiter=500, tol=1e-9)
            score_opt = _score(opt_pts)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = opt_pts.copy()
        except Exception:
            pass

    # Strategy 3: Optimize from the regular polygon directly
    if time.time() > deadline - 15:
        pass
    else:
        try:
            opt_reg = _optimize_from_start(pts_reg, maxiter=1000, tol=1e-10)
            score_reg_opt = _score(opt_reg)
            if score_reg_opt > best_score:
                best_score = score_reg_opt
                best_pts = opt_reg.copy()
        except Exception:
            pass

    # Strategy 4: Polish the best result with higher precision
    if best_pts is not None and time.time() > deadline - 10:
        pass
    elif best_pts is not None:
        try:
            polished = _optimize_from_start(best_pts, maxiter=2000, tol=1e-12)
            score_polished = _score(polished)
            if score_polished > best_score:
                best_score = score_polished
                best_pts = polished.copy()
        except Exception:
            pass

    # If we still have time, try a few more random starts
    if best_pts is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Start from a perturbed version of the best
            perturbed = best_pts + rng.normal(0, 0.01, best_pts.shape)
            try:
                opt_pert = _optimize_from_start(perturbed, maxiter=300, tol=1e-9)
                score_pert = _score(opt_pert)
                if score_pert > best_score:
                    best_score = score_pert
                    best_pts = opt_pert.copy()
            except Exception:
                pass

    # Final polish if we have time
    if best_pts is not None and time.time() < deadline - 5:
        try:
            final_polish = _optimize_from_start(best_pts, maxiter=1000, tol=1e-12)
            score_final = _score(final_polish)
            if score_final > best_score:
                best_pts = final_polish.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback to regular polygon
        best_pts = pts_reg

    return best_pts
