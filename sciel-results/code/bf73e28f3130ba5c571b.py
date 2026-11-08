import numpy as np
import time
from scipy.optimize import minimize

N = 13

def _triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    return 0.5 * abs(np.cross(q - p, r - p))

def _min_triangle_area(points):
    """Minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Area of convex hull of points."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return hull.volume

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    return min_area / hull_area

def _objective_angles(angles):
    """Objective for angle optimization (negative score)."""
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    score = _score(points)
    return -score

def _regular_polygon_angles():
    """Angles for regular N-gon."""
    return np.linspace(0, 2 * np.pi, N, endpoint=False)

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to hull area.
    Optimizes angles of points on unit circle starting from regular polygon.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Start with regular polygon angles
    best_angles = _regular_polygon_angles()
    best_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    best_score = _score(best_points)
    
    # Local optimization on angles
    try:
        result = minimize(
            _objective_angles,
            best_angles.copy(),
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-7,
                'fatol': 1e-9,
                'adaptive': True
            }
        )
        
        if result.fun < -best_score:
            opt_angles = result.x
            opt_points = np.column_stack((np.cos(opt_angles), np.sin(opt_angles)))
            opt_score = _score(opt_points)
            if opt_score > best_score:
                best_angles = opt_angles
                best_points = opt_points
                best_score = opt_score
                
    except Exception:
        pass
    
    if time.time() - start_time > deadline:
        return best_points
    
    # Try a few perturbed starts
    rng = np.random.RandomState(123)
    for attempt in range(3):
        if time.time() - start_time > deadline:
            break
        
        noise = rng.randn(N) * 0.05
        perturbed_angles = best_angles + noise
        # Wrap angles to [0, 2*pi)
        perturbed_angles = perturbed_angles % (2 * np.pi)
        
        try:
            result = minimize(
                _objective_angles,
                perturbed_angles,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_score:
                opt_angles = result.x
                opt_points = np.column_stack((np.cos(opt_angles), np.sin(opt_angles)))
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_angles = opt_angles
                    best_points = opt_points
                    best_score = opt_score
                    
        except Exception:
            pass
    
    return best_points
