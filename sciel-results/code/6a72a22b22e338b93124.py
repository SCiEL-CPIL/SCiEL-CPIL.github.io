"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
        
    # Center points to remove translation invariance
    points = points - np.mean(points, axis=0)
    
    # Check for degenerate points
    if np.allclose(points, 0):
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
        
        # Scale to unit area
        scale = np.sqrt(hull_area)
        scaled_points = points / scale
        
        # Calculate min triangle area
        min_area = float('inf')
        for i in range(n - 2):
            p1 = scaled_points[i]
            for j in range(i + 1, n - 1):
                p2 = scaled_points[j]
                v1 = p2 - p1
                for k in range(j + 1, n):
                    p3 = scaled_points[k]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
                        
        return min_area
    except Exception:
        return 0.0


def _objective(flat_points):
    """
    Objective function for minimization.
    We want to maximize score, so we minimize -score.
    """
    points = flat_points.reshape(-1, 2)
    s = _score(points)
    if s == 0.0:
        return 1e6 # Penalty
    return -s


def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin before 120s limit
    
    best_score = 0.0
    best_points = None
    
    # Initial guesses
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    inits.append(np.column_stack((x, y)).flatten())
    
    # 2-5. Perturbed regular polygons (small noise)
    for i in range(4):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.05, 0.05, N)
        radii = 1.0 + rng.uniform(-0.05, 0.05, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)).flatten())
        
    # 6-10. Random points in unit square
    for i in range(5):
        pts = rng.random((N, 2))
        inits.append(pts.flatten())
        
    # 11-15. Random convex-ish points (larger radius perturbation)
    for i in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + rng.uniform(-0.2, 0.2, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)).flatten())
        
    # 16-20. More random perturbations of regular polygon
    for i in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + rng.uniform(-0.1, 0.1, N)
        # Randomly swap some angles to break symmetry? No, keep order.
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)).flatten())

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, init, method='Nelder-Mead', 
                           options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 1e6:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(-1, 2)
                    
        except Exception:
            continue

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
# EVOLVE-BLOCK-END
