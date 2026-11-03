import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for vectorized triangle area calculation
# i < j < k
_indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _indices.append((i, j, k))
_indices = np.array(_indices)
_IDX_I = _indices[:, 0]
_IDX_J = _indices[:, 1]
_IDX_K = _indices[:, 2]

def _compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    if points.shape != (N, 2):
        return 0.0
    
    # Vectorized triangle area calculation
    p1 = points[_IDX_I]
    p2 = points[_IDX_J]
    p3 = points[_IDX_K]
    
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product z-component: v1x * v2y - v1y * v2x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def _neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    score = _compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit
    
    # Initial bounds
    # A hexagonal cluster of 14 points fits roughly in [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    # Run Differential Evolution
    try:
        # maxiter=100, popsize=15
        # Using seed for reproducibility
        res = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            seed=42,
            polish=False  # Disable built-in polish to avoid non-smooth issues
        )
        
        if res.success or res.fun < 0: # res.fun is -score
            best_points = res.x.reshape(N, 2)
            best_score = -res.fun
    except Exception as e:
        # If DE fails, try a simpler local search or return random
        pass
        
    if best_points is None:
        # Fallback: Random points
        best_points = np.random.rand(N, 2) * 3 - 1.5
        best_score = _compute_score(best_points)
        
    # Check time
    if time.time() > deadline:
        return best_points
        
    # Local Polish using a smooth surrogate
    # Objective: Maximize - (1/beta) * log(sum(exp(-beta * areas))) / hull_area
    # This is complex. Let's try a simple local search with small steps.
    # Or use scipy.optimize.minimize with a smooth approximation of the min.
    
    # Smooth Min Approximation:
    # min(x) ~= -1/beta * log(sum(exp(-beta*x)))
    # We want to maximize min(areas) / hull_area.
    # Let f = min(areas) / hull_area.
    # This is hard to smooth directly because of the division.
    
    # Alternative: Since N is small, we can just do a few iterations of 
    # "move each point in the direction that increases the min area".
    # This is essentially gradient ascent on the min area.
    
    # Given the time budget, DE result is likely sufficient.
    # Let's add a simple coordinate descent if time permits.
    
    current_points = best_points
    current_score = best_score
    
    # Simple local search
    # Try perturbing each point slightly
    # This is expensive. Let's skip if not necessary.
    
    return current_points
