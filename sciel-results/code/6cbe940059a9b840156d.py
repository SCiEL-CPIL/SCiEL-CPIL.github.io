"""
Strategy: Use Differential Evolution (global stochastic search) to maximize the 
minimum triangle area relative to the convex hull area. 
This avoids the local minima traps of the Nelder-Mead method used in the seed.
We initialize around the regular 14-gon, which is a strong candidate, and let 
DE explore the neighborhood and potentially find better non-regular convex configurations.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = points.shape[0]
    min_area = np.inf
    # Vectorized computation for speed
    # i, j, k indices
    # We can use broadcasting or loops. For N=14, C(14,3)=364 triangles.
    # Loops are fast enough for 364 triangles.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or degenerate
        return 1e-10

def _score(points):
    """Score is min_triangle_area / convex_hull_area"""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(flat_points):
    """Objective for minimizer - exact score."""
    points = flat_points.reshape(N, 2)
    return -_score(points)

def _generate_regular_polygon():
    """Generate a regular 14-gon centered at origin with circumradius 1."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (actually 150s kill, so 115 is safe)
    
    # Baseline: Regular 14-gon
    best_points = _generate_regular_polygon()
    best_score = _score(best_points)
    
    # Objective function for Differential Evolution
    def obj(x):
        if time.time() > deadline:
            return 1e10
        points = x.reshape(N, 2)
        # Optional: Normalize points to prevent drift?
        # The score is scale-invariant, so drift is not a problem for the score,
        # but DE might wander. We can re-center and scale in the objective?
        # No, let DE handle the scale. The regular polygon is at radius 1.
        return -_score(points)
    
    # Bounds for DE
    # Regular polygon is in [-1, 1] x [-1, 1].
    # Allow a bit of space for optimization.
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Run Differential Evolution
    # DE is slow but global.
    # popsize: 15-30 is typical.
    # maxiter: Limit by time.
    try:
        # Use a smaller popsize and fewer iterations to fit in time, 
        # or rely on the time check in obj.
        res = differential_evolution(
            obj, 
            bounds, 
            maxiter=500, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=False  # Disable Nelder-Mead polish to save time and avoid local traps
        )
        
        if time.time() < deadline:
            candidate_points = res.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
                
    except Exception:
        pass
    
    # If time remains, try a local polish from the best point found by DE
    if time.time() < deadline - 10:
        try:
            # Use Nelder-Mead for a quick local refinement
            res_local = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if time.time() < deadline:
                local_points = res_local.x.reshape(N, 2)
                local_score = _score(local_points)
                if local_score > best_score:
                    best_score = local_score
                    best_points = local_points
        except Exception:
            pass
            
    # Final fallback
    if best_points is None:
        return _generate_regular_polygon()
        
    return best_points
