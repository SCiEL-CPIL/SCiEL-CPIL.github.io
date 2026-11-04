import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def _triangle_area(p1, p2, p3):
    """
    Compute the area of the triangle formed by points p1, p2, p3.
    """
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def _min_triangle_area(points):
    """
    Compute the minimum area of all triangles formed by triples of points.
    """
    n = len(points)
    min_area = float('inf')
    # Precompute all pairs? No, just iterate. 364 triangles is small.
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                area = _triangle_area(p1, p2, p3)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Normalizes the points to have unit convex hull area.
    """
    # Translate to centroid
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Compute hull area
    hull_area = _convex_hull_area(points_centered)
    if hull_area < 1e-9:
        return -1e9 # Penalty for degenerate hull
        
    # Scale to unit hull area
    # Area scales with r^2, so r scales with 1/sqrt(area)
    scale_factor = 1.0 / np.sqrt(hull_area)
    points_scaled = points_centered * scale_factor
    
    # Compute min triangle area on scaled points
    min_area = _min_triangle_area(points_scaled)
    
    return min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = -1e9
    best_points = initial_points
    
    # Objective function for optimizer (we want to maximize score, so minimize -score)
    def objective(x):
        points = x.reshape(N, 2)
        s = _score(points)
        return -s

    # Run a few restarts with perturbations
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Add small random noise to initial points
        noise = np.random.normal(0, 0.1, (N, 2))
        x0 = (initial_points + noise).flatten()
        
        # Optimize
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000})
            
            if res.success or res.fun < -best_score:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception as e:
            pass
            
    return best_points
