"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1e-10  # Avoid division by zero


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _objective(points_1d):
    """
    Objective function for minimization.
    Returns negative of the score (min triangle area / hull area).
    """
    points = points_1d.reshape(N, 2)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1.0  # Large value (bad) since we are minimizing
    min_area = _min_triangle_area(points)
    score = min_area / hull_area
    return -score


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Use Differential Evolution for global search, followed by local polishing
    with Nelder-Mead.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave buffer for safety
    
    # Bounds: Keep points within [-1, 1] x [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Initial guess: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = init_points.flatten()
    
    best_val = _objective(x0)
    best_x = x0.copy()
    
    # 1. Global Search with Differential Evolution
    try:
        de_result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=0.01, 
            seed=42,
            init='sobol',
            polish=False
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x
    except Exception:
        pass
        
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return best_x.reshape(N, 2)
    
    # 2. Local Polish with Nelder-Mead
    try:
        local_result = minimize(
            _objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if local_result.fun < best_val:
            best_val = local_result.fun
            best_x = local_result.x
    except Exception:
        pass
        
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
