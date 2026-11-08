"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _objective(params):
    """
    Objective function to minimize (negative of the score).
    params: array of shape (2*N,)
    """
    points = params.reshape(-1, 2)
    
    # Normalize points: center at origin and fix scale
    # This ensures the problem is well-posed and prevents scaling drift
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Fix scale: sum of squared distances from origin = N
    norm_sq = np.sum(points**2)
    if norm_sq < 1e-10:
        return 1e6
    points = points * np.sqrt(N / norm_sq)
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 1e6
        
    if hull_area < 1e-10:
        return 1e6
        
    # Calculate Minimum Triangle Area
    min_area = float('inf')
    # Iterate over all triples
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    
    score = min_area / hull_area
    return -score

def construct():
    """
    Uses numerical optimization to maximize the minimum triangle area 
    relative to the convex hull area.
    """
    # Initial guess: Regular N-gon
    angles = 2 * np.pi * np.arange(N) / N
    x = np.cos(angles)
    y = np.sin(angles)
    points_init = np.column_stack((x, y)).flatten()
    
    # Run optimization
    # Nelder-Mead is derivative-free and robust for this small dimension
    res = minimize(_objective, points_init, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    optimized_points = res.x.reshape(-1, 2)
    
    return optimized_points
# EVOLVE-BLOCK-END
