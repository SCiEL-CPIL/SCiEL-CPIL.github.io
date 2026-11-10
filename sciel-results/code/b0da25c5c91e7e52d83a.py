# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    points: (n, 2) array
    """
    n = points.shape[0]
    min_area = float('inf')
    # Vectorized approach for speed if possible, but n=13 is small
    # 13 choose 3 = 286 triangles
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 0.0

def _objective(params):
    """
    Objective function for minimization.
    Minimizes - (A_min / A_hull)
    """
    points = params.reshape(13, 2)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 1e10
        
    a_hull = _hull_area(points)
    if a_hull < 1e-9:
        return 1e10
        
    a_min = _min_triangle_area(points)
    
    # If any area is zero (collinear), score is 0, so we return a large positive value
    if a_min < 1e-9:
        return 1e10
        
    ratio = a_min / a_hull
    return -ratio

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Initial guesses
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Grid-like perturbation
    rng = np.random.default_rng(42)
    init_grid = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.5],
        [0.2, 0.8], [0.8, 0.8], [0.5, 0.2],
        [0.5, 0.8], [0.2, 0.5], [0.8, 0.5],
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]
    ])
    
    # 3. Random points
    init_random = rng.random((n, 2)) * 0.8 + 0.1
    
    initial_guesses = [init_regular, init_grid, init_random]
    
    # Multi-start local optimization
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        params = guess.flatten()
        
        # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness okay
        try:
            res = __import__('scipy').optimize.minimize(
                _objective, 
                params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(13, 2)
        except Exception:
            pass

    # If we have time, do a second pass with Differential Evolution or more local starts
    if time.time() < deadline - 60 and best_points is not None:
        # Refine the best found point with a tighter tolerance
        try:
            res_refine = __import__('scipy').optimize.minimize(
                _objective, 
                best_points.flatten(), 
                method='Powell', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_refine.fun < best_score:
                best_score = res_refine.fun
                best_points = res_refine.x.reshape(13, 2)
        except Exception:
            pass

    # If no valid points were found (should not happen), return regular polygon
    if best_points is None:
        return init_regular

    return best_points

# EVOLVE-BLOCK-END
