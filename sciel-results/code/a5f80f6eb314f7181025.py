"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    # Vectorized calculation for speed
    # i, j, k triples
    # This loop is O(N^3) but N=13 is small (286 triangles)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    ma = _min_triangle_area(points)
    ha = _convex_hull_area(points)
    if ha < 1e-10:
        return -1e9
    return ma / ha

def _objective_flat(params):
    """
    Objective function for flat array of 2*N parameters.
    Returns negative score to minimize.
    """
    points = params.reshape((N, 2))
    # Center the points to remove translation invariance
    # This helps the optimizer by keeping points near origin
    points = points - np.mean(points, axis=0)
    
    s = _score(points)
    # Penalize if points are too close to each other or hull is degenerate
    # The score function handles this, but we ensure finite return
    if not np.isfinite(s):
        return 1e9
    return -s

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Approach: Direct coordinate optimization.
    1. Global search using Differential Evolution on 26 coordinates.
    2. Local refinement using Powell's method (derivative-free, robust).
    """
    start_time = time.time()
    timeout = 115.0
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0_poly = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_params = x0_poly
    best_score = _score(x0_poly.reshape((N, 2)))
    
    # Phase 1: Differential Evolution
    # Bounds: [-2, 2] for x and y
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    try:
        if time.time() - start_time < timeout - 60:
            de_res = differential_evolution(
                _objective_flat,
                bounds,
                seed=42,
                maxiter=50, # Keep iterations low to allow time for refinement
                popsize=20,
                tol=1e-5,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=False, # We do our own refinement
                init='sobol'
            )
            
            if de_res.fun < -best_score: # fun is -score
                best_params = de_res.x
                best_score = -de_res.fun
    except Exception:
        pass
    
    # Phase 2: Local Refinement with Powell
    # Powell is good for non-smooth functions and doesn't require gradients
    if time.time() - start_time < timeout - 10:
        try:
            res = minimize(
                _objective_flat,
                best_params,
                method='Powell',
                options={
                    'maxiter': 1000,
                    'maxfev': 5000,
                    'xtol': 1e-8,
                    'ftol': 1e-8
                }
            )
            
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass
            
    # Fallback if everything failed
    if best_score <= 0:
        best_params = x0_poly
        best_score = _score(x0_poly.reshape((N, 2)))
        
    points = best_params.reshape((N, 2))
    
    # Center points
    points = points - np.mean(points, axis=0)
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        points = points * scale
        
    return points
# EVOLVE-BLOCK-END
