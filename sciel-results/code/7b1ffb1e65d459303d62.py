"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize, differential_evolution


def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return 0.0
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _score_from_points(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _grid_base_points():
    """Generate base 3x5 grid points with 2 removed to get 13 points.
    Remove two corner-adjacent points to create a more balanced structure."""
    # 3 rows x 5 cols grid
    base = []
    for r in range(3):
        for c in range(5):
            base.append([c, r])
    # Remove two points: (0,0) and (4,2) to get 13 points
    # Actually let's remove (1,0) and (3,2) for better distribution
    remove_indices = [1, 12]  # (1,0) and (3,2) in row-major order
    indices_to_keep = [i for i in range(15) if i not in remove_indices]
    return np.array([base[i] for i in indices_to_keep], dtype=float)


def _apply_affine(points, params):
    """Apply affine transformation: [[a,b],[c,d]] @ point"""
    a, b, c, d = params
    M = np.array([[a, b], [c, d]])
    return points @ M.T


def _objective(params, base_points):
    """Negative score for minimization."""
    points = _apply_affine(base_points, params)
    return -_score_from_points(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    base_points = _grid_base_points()
    
    best_score = 0.0
    best_params = np.array([1.0, 0.0, 0.0, 1.0])
    
    # Try several initial parameter sets
    initial_candidates = [
        [1.0, 0.0, 0.0, 1.0],      # identity
        [1.0, 0.0, 0.0, 0.5],      # vertical compression
        [1.0, 0.0, 0.0, 2.0],      # vertical stretch
        [0.5, 0.0, 0.0, 1.0],      # horizontal compression
        [2.0, 0.0, 0.0, 1.0],      # horizontal stretch
        [1.0, 0.5, 0.0, 1.0],      # shear
        [1.0, -0.5, 0.0, 1.0],     # negative shear
        [0.7, 0.3, 0.3, 0.7],      # rotated-ish
        [1.2, 0.2, 0.2, 1.2],      # slight rotation
    ]
    
    for init in initial_candidates:
        if time.time() > deadline:
            break
        try:
            result = minimize(_objective, init, args=(base_points,), method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Also try differential evolution on a smaller range
    if time.time() < deadline - 30:
        try:
            bounds = [(-2, 3), (-2, 2), (-2, 2), (-2, 3)]
            de_result = differential_evolution(_objective, bounds, args=(base_points,),
                                              maxiter=200, popsize=15, tol=1e-8,
                                              seed=42, polish=True)
            score = -de_result.fun
            if score > best_score:
                best_score = score
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Final local refinement from best found
    if time.time() < deadline - 10 and best_score > 0:
        try:
            result = minimize(_objective, best_params, args=(base_points,), method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    final_points = _apply_affine(base_points, best_params)
    
    # Verify we have 13 distinct points
    if len(final_points) != N:
        # Fallback to random
        return np.random.rand(N, 2)
    
    return final_points

# EVOLVE-BLOCK-END
