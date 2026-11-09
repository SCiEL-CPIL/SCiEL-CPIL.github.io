import numpy as np
from scipy.optimize import minimize
import time

N = 14

def _compute_score(angles):
    """Compute the score: min triangle area / convex hull area for points on unit circle."""
    angles = np.sort(angles)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Compute convex hull area using shoelace formula (points are in convex position on circle)
    n = len(points)
    area_hull = 0.0
    for i in range(n):
        j = (i + 1) % n
        area_hull += points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1]
    area_hull = abs(area_hull) / 2.0
    
    if area_hull < 1e-10:
        return -1.0  # Degenerate
    
    # Compute minimum triangle area
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    
    return min_area / area_hull

def _negative_score(angles):
    return -_compute_score(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Initialize with regular polygon angles
    angles_init = 2 * np.pi * np.arange(N) / N
    
    best_score = _compute_score(angles_init)
    best_angles = angles_init.copy()
    
    # Try multiple local optimizations from different starting points
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb the initial angles
        if trial == 0:
            angles = angles_init.copy()
        else:
            # Small random perturbation
            angles = angles_init + rng.randn(N) * 0.1
            angles = np.sort(angles)
        
        result = minimize(_negative_score, angles, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_angles = np.sort(result.x)
    
    # Final result
    angles = best_angles
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return points
