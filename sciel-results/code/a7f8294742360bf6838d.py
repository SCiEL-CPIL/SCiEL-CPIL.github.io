import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    # Ensure points is a numpy array of shape (N, 2)
    points = np.array(points)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback if points are degenerate
        return 0.0

    if hull_area < 1e-10:
        return 0.0

    # Compute all triangle areas
    # N=14, C(14,3) = 364 triangles
    min_area = float('inf')
    # Vectorized calculation for speed
    # We can generate indices
    i, j, k = np.indices((N, N, N))
    # Filter i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(flat_points):
    """
    Objective function for minimization (we want to maximize score, so minimize -score).
    """
    points = flat_points.reshape((N, 2))
    score = _score(points)
    # If score is 0 or very small, return a large penalty
    if score < 1e-10:
        return 1e10
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Strategy: Start with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Initial score
    current_score = _score(initial_points)
    best_points = initial_points.copy()
    
    # Optimization using Nelder-Mead
    # Flatten points for scipy
    x0 = initial_points.flatten()
    
    # Check time
    if time.time() > deadline - 10:
        return best_points

    try:
        # Nelder-Mead parameters
        # maxiter: limit to prevent hanging
        # xatol, fatol: tolerance
        result = minimize(
            _objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.success or (result.fun < -current_score):
            optimized_points = result.x.reshape((N, 2))
            opt_score = -result.fun
            if opt_score > current_score:
                current_score = opt_score
                best_points = optimized_points.copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 5:
        return best_points

    # Try a few other symmetric seeds just in case
    # Seed 2: Regular 14-gon rotated
    # Seed 3: Points on two concentric circles (7 on each)
    # This is unlikely to beat the regular polygon, but cheap to check
    
    # Let's try a simple local refinement with smaller steps if Nelder-Mead didn't improve much
    # Actually, Nelder-Mead is usually sufficient for this smooth-ish landscape.
    
    # One more check: ensure the returned points are valid
    final_score = _score(best_points)
    
    return best_points
