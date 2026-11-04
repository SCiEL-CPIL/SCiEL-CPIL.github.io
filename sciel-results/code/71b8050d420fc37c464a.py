import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    """
    # Ensure points are 2D
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0 # Degenerate hull
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    # There are C(13, 3) = 286 triangles
    min_area = float('inf')
    n = len(points)
    # Vectorized calculation for all triples is possible but 286 is small.
    # We can loop for clarity or use combinations.
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * | (p2 - p1) x (p3 - p1) |
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        if area < min_area:
            min_area = area
            
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    # Initialize with a regular 13-gon
    # This is a good starting point as it is symmetric and convex.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    x0 = np.column_stack((x, y)).flatten()
    
    # Define the objective function for the optimizer
    def objective(x):
        points = x.reshape(-1, 2)
        return _score(points)
        
    # Use Nelder-Mead for local optimization
    # maxiter is set to ensure it finishes within time limits
    # N=13 means 26 parameters. Nelder-Mead is robust here.
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    # Extract the optimized points
    opt_points = res.x.reshape(-1, 2)
    
    # Return the points
    return opt_points
# EVOLVE-BLOCK-END
