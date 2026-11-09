# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # Safety margin before 630s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    # For n=13, C(13,3) = 286.
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _objective(x):
    """
    Objective function to minimize (negative of the score).
    Score = A_min / A_hull
    """
    points = x.reshape(13, 2)
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return -(a_min / a_hull)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Try multiple random restarts
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > _DEADLINE:
            break
            
        # Random initialization in [0, 1] x [0, 1]
        x0 = rng.random((n, 2)).flatten()
        
        # Use Nelder-Mead to minimize the negative score
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(n, 2)
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to random if optimization failed completely
        best_points = rng.random((n, 2))
        
    return best_points

# EVOLVE-BLOCK-END
