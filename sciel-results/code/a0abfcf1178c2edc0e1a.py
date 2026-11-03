"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START

def _triangle_area(p1, p2, p3):
    """Area of triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Find minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    if min_area == 0:
        return 0.0
    
    return min_area / hull_area


def _neg_objective(x):
    """Negative of the score for minimization."""
    points = x.reshape(-1, 2)
    score = _min_triangle_area_over_hull(points)
    return -score


def construct():
    """
    Place 14 points in a planar convex region normalized to unit area 
    to maximize the minimum area over all triples.
    Returns array of shape (14, 2).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -1.0
    
    # Initial symmetric configuration: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    hull = ConvexHull(initial)
    hull_area = hull.volume
    if hull_area > 0:
        initial = initial / np.sqrt(hull_area)
    
    best_points = initial.copy()
    best_score = _min_triangle_area_over_hull(best_points)
    
    # Local optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    
    n_attempts = 20
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            x0 = initial.flatten()
        else:
            # Random perturbation around a circle or random points
            if attempt % 2 == 0:
                # Perturbed regular polygon
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
                radii = 1.0 + rng.uniform(-0.1, 0.1, N)
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            else:
                # Random points in unit square
                x0 = rng.uniform(0, 1, (N, 2)).flatten()
        
        try:
            result = minimize(_neg_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.success or result.fun < -best_score:
                new_points = result.x.reshape(-1, 2)
                new_score = _min_triangle_area_over_hull(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            continue
    
    return best_points

# EVOLVE-BLOCK-END
