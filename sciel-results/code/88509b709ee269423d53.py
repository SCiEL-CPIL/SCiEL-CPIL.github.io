import numpy as np
from scipy.optimize import minimize
import time

N = 13

def _triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    """Compute area of convex hull using scipy."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 0.0

def _score(points):
    """Compute the minimum triangle area divided by convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    
    return min_area / hull_area

def _score_from_angles(angles):
    """Convert angles to points on unit circle and compute score."""
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    return _score(points)

def _negative_score_from_angles(angles):
    """Negative score for minimization."""
    return -_score_from_angles(angles)

def construct():
    """Place N points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with regular polygon
    angles0 = 2 * np.pi * np.arange(N) / N
    
    # Try multiple initializations and optimizations
    best_score = -float('inf')
    best_angles = angles0.copy()
    
    # Optimization 1: Start from regular polygon
    result = minimize(_negative_score_from_angles, angles0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    if time.time() > deadline:
        # Return best found so far
        points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
        return points
    
    if result.fun < -best_score:
        best_score = -result.fun
        best_angles = result.x.copy()
    
    # Try different initializations: perturb the regular polygon
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Create perturbed initial angles
        angles_init = angles0.copy()
        noise = np.random.randn(N) * 0.1
        # Ensure angles are sorted and within [0, 2π)
        angles_init = angles_init + noise
        angles_init = np.sort(angles_init) % (2 * np.pi)
        
        result = minimize(_negative_score_from_angles, angles_init, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if time.time() > deadline:
            break
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_angles = result.x.copy()
    
    # Try L-BFGS-B with bounds
    if time.time() < deadline:
        bounds = [(0, 2 * np.pi)] * N
        result = minimize(_negative_score_from_angles, best_angles, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15})
        if result.fun < -best_score:
            best_score = -result.fun
            best_angles = result.x.copy()
    
    # Final: ensure angles are sorted for consistent output
    best_angles = np.sort(best_angles) % (2 * np.pi)
    
    # Convert to points
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    points = np.column_stack((x, y))
    
    return points
