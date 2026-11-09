import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """Compute minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _points_from_params(params):
    """Convert parameter vector to point coordinates.
    params: [r_0, ..., r_13, theta_0, ..., theta_13]
    First 14 are radii, next 14 are angular offsets from regular polygon.
    """
    n = N
    base_angles = 2 * np.pi * np.arange(n) / n
    radii = params[:n]
    angle_offsets = params[n:2*n]
    angles = base_angles + angle_offsets
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _negative_score(params):
    """Negative score for minimization."""
    points = _points_from_params(params)
    return -_score(points)

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with regular 14-gon
    base_angles = 2 * np.pi * np.arange(N) / N
    init_params = np.zeros(2 * N)
    init_params[:N] = 1.0  # unit radii
    init_params[N:] = 0.0   # zero angular offsets
    
    # Phase 1: Differential Evolution
    bounds = [
        [(0.5, 1.5)] * N +  # radii
        [(-np.pi / N, np.pi / N)] * N  # angular offsets
    ]
    
    try:
        de_result = differential_evolution(
            _negative_score,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=False,
            init='sobol'
        )
        
        if time.time() > deadline - 30:
            # Not enough time for refinement, return DE result
            best_params = de_result.x
        else:
            # Phase 2: Nelder-Mead refinement
            nm_result = minimize(
                _negative_score,
                de_result.x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            best_params = nm_result.x
    except Exception:
        best_params = init_params
    
    points = _points_from_params(best_params)
    
    # Ensure points are in convex position - if not, fall back to regular polygon
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            # Some points are interior; use regular polygon
            points = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    except Exception:
        points = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    return points

# EVOLVE-BLOCK-START
# (The entire construct function and helpers are within the evolve block scope)
# EVOLVE-BLOCK-END
