"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
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
        return hull.area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _generate_initial_points(n, seed):
    """Generate initial points with a mix of strategies."""
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Regular polygon with perturbations
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = 1.0 + 0.1 * rng.randn(n)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Strategy 2: Random points in unit disk
    angles2 = rng.uniform(0, 2*np.pi, n)
    radii2 = np.sqrt(rng.uniform(0, 1, n))
    points2 = np.column_stack([radii2 * np.cos(angles2), radii2 * np.sin(angles2)])
    
    # Strategy 3: Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    angles3 = np.arange(n) * golden_angle
    radii3 = np.sqrt(np.arange(n) / n)
    points3 = np.column_stack([radii3 * np.cos(angles3), radii3 * np.sin(angles3)])
    
    # Return a list of initial points
    return [points, points2, points3]

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s safety margin before 160s limit
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial points
    initial_points_list = _generate_initial_points(N, seed=42)
    
    for init_idx, init_points in enumerate(initial_points_list):
        if time.time() > deadline:
            break
        
        # Try multiple optimization methods
        for method in ['L-BFGS-B', 'Nelder-Mead', 'Powell']:
            if time.time() > deadline:
                break
            
            def objective(params):
                # Reshape params to (n, 2)
                pts = params.reshape(-1, 2)
                return -_score(pts)
            
            # Initial guess
            x0 = init_points.flatten()
            
            # Bounds: keep points in a reasonable range
            bounds = [(-2, 2)] * (2 * N)
            
            try:
                result = minimize(objective, x0, method=method, bounds=bounds,
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if result.success or result.fun < 0:
                    opt_points = result.x.reshape(-1, 2)
                    opt_score = _score(opt_points)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
            except Exception:
                pass
    
    # If we didn't find a good solution, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Final polish: try to improve with a tighter optimization
    if time.time() < deadline - 10:
        def objective(params):
            pts = params.reshape(-1, 2)
            return -_score(pts)
        
        x0 = best_points.flatten()
        bounds = [(-2, 2)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.fun < 0:
                polish_points = result.x.reshape(-1, 2)
                polish_score = _score(polish_points)
                
                if polish_score > best_score:
                    best_points = polish_points.copy()
                    best_score = polish_score
        except Exception:
            pass
    
    # Ensure the points are valid (non-degenerate)
    hull_area = _convex_hull_area(best_points)
    if hull_area <= 0:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
