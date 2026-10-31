"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    mta = _min_triangle_area(points)
    return mta / ha

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Initial guess: points on a circle (good symmetric starting point)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Also try a regular polygon-like arrangement
    best_points = x0.copy()
    best_score = _score(best_points)
    
    def objective(flat_pts):
        pts = flat_pts.reshape(N, 2)
        ha = _hull_area(pts)
        if ha < 1e-10:
            return -1e10
        mta = _min_triangle_area(pts)
        # We want to maximize min_triangle_area / hull_area
        # So minimize -min_triangle_area / hull_area
        return -mta / ha
    
    # Try multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    
    n_restarts = 0
    while time.time() < deadline - 10.0:
        n_restarts += 1
        
        # Generate a random starting point
        if n_restarts == 1:
            x0_try = x0.copy()
        else:
            # Mix of circle-based and random perturbations
            if n_restarts % 3 == 0:
                # Regular polygon with slight perturbation
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
                x0_try = np.column_stack([np.cos(angles), np.sin(angles)])
            elif n_restarts % 3 == 1:
                # Random points in unit square
                x0_try = rng.rand(N, 2)
            else:
                # Points on a circle with random radii
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.2, 0.2, N)
                radii = 0.8 + 0.2 * rng.rand(N)
                x0_try = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Local optimization
        try:
            result = minimize(
                objective,
                x0_try.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            opt_pts = result.x.reshape(N, 2)
            s = _score(opt_pts)
            
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Final polish: try to improve the best solution with tighter tolerances
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            opt_pts = result.x.reshape(N, 2)
            s = _score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
