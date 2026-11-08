import numpy as np
import time
from scipy.optimize import minimize

N = 13

def _triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    # Manual calculation to avoid np.cross dimensionality issues
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return 0.5 * abs(dx1 * dy2 - dy1 * dx2)

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
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If hull fails (e.g., collinear points), return 0 to invalidate the score
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _objective_angles(angles):
    """Objective for angle optimization (negative score)."""
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    score = _score(points)
    return -score

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to hull area.
    Uses local optimization on angles starting from regular polygon.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start with regular polygon angles
    best_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    best_score = _score(best_points)
    
    # Local optimization on angles
    try:
        result = minimize(
            _objective_angles,
            best_angles.copy(),
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
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
    rng = np.random.RandomState(42)
    for attempt in range(4):
        if time.time() - start_time > deadline:
            break
        
        noise = rng.randn(N) * 0.1
        perturbed_angles = best_angles + noise
        perturbed_angles = perturbed_angles % (2 * np.pi)
        
        try:
            result = minimize(
                _objective_angles,
                perturbed_angles,
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
    
    return best_points
