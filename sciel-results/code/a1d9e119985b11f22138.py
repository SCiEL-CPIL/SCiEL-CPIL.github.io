"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Calculate areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.empty((n, n, n))
    # We only need i < j < k, but for vectorization simplicity or speed, 
    # we can compute all and take the min, but that's wasteful.
    # Let's just loop over triples since N=14 is small.
    min_area = np.inf
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Area = 0.5 * |cross(v1, v2)|
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or fewer than 3, area is 0
        return 0.0

def _score(points):
    """Calculate the normalized minimum triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = _triangle_areas(points)
    return min_tri_area / hull_area

def _negative_smooth_score(points):
    """
    Objective function for minimization.
    We want to maximize the minimum area.
    We use a smooth approximation of the minimum to allow gradient-based methods,
    or we can just use Nelder-Mead on the raw min.
    Nelder-Mead is robust to non-smoothness.
    Let's use the raw min for accuracy in the final step, but for optimization,
    a smoothed version might converge faster.
    However, for N=14, the raw min is fast enough for Nelder-Mead.
    """
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial guess: Regular N-gon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Optimization using Nelder-Mead
    # We flatten the points to a vector of length 2N
    x0 = best_points.flatten()
    
    # Options for Nelder-Mead
    options = {
        'maxiter': 5000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        result = minimize(_negative_smooth_score, x0, method='Nelder-Mead', options=options)
        if result.success or result.fun < -best_score:
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass

    # Check time
    if time.time() < deadline:
        # Try a few random perturbations from the regular polygon
        # to escape local minima if any
        for i in range(5):
            if time.time() > deadline:
                break
            noise = np.random.normal(0, 0.1, (N, 2))
            start_points = best_points + noise
            x0 = start_points.flatten()
            try:
                result = minimize(_negative_smooth_score, x0, method='Nelder-Mead', options=options)
                if result.fun < -best_score:
                    candidate_points = result.x.reshape(N, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_points = candidate_points
                        best_score = candidate_score
            except Exception:
                pass

    return best_points

# EVOLVE-BLOCK-END
