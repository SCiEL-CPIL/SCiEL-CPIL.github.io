"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
        
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
    Higher is better.
    """
    if len(points) < 3:
        return 0.0
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # This is smooth and approximates min(a) as beta -> inf
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _get_ring_points(r1, theta1, r2, theta2):
    """
    Generate 14 points with 7-fold rotational symmetry.
    Two rings of 7 points each.
    Ring 1: radius r1, offset theta1
    Ring 2: radius r2, offset theta2
    """
    step = 2 * np.pi / 7
    angles = np.arange(7) * step
    points = []
    
    # Ring 1
    for a in angles:
        x = r1 * np.cos(a + theta1)
        y = r1 * np.sin(a + theta1)
        points.append([x, y])
        
    # Ring 2
    for a in angles:
        x = r2 * np.cos(a + theta2)
        y = r2 * np.sin(a + theta2)
        points.append([x, y])
        
    return np.array(points)

def _score_ring_params(params):
    """
    Objective function for differential evolution.
    params: [r1, theta1, r2, theta2]
    Returns negative score for minimization.
    """
    r1, theta1, r2, theta2 = params
    
    # Ensure distinct points and valid geometry
    # r1, r2 should be positive
    if r1 < 1e-6 or r2 < 1e-6:
        return 0.0
        
    points = _get_ring_points(r1, theta1, r2, theta2)
    
    # Check for duplicate points (within tolerance)
    # Simple check: if any two points are very close
    # This is O(N^2) but N=14 is small
    # However, for speed in DE, we might skip this or do a very cheap check
    # Let's rely on the score function returning 0 if hull fails or area is tiny
    
    score = _calculate_score(points)
    return -score

def _soft_score_ring_params(params, beta=50.0):
    """
    Smooth surrogate objective for ring parameters using soft-min.
    Returns negative soft score for minimization.
    """
    r1, theta1, r2, theta2 = params
    
    if r1 < 1e-6 or r2 < 1e-6:
        return 0.0
        
    points = _get_ring_points(r1, theta1, r2, theta2)
    
    # Use soft min score
    soft_score = _soft_min_score(points, beta)
    return -soft_score

def _coordinate_polish(points, time_budget, deadline):
    """
    Polish all 28 coordinates directly using L-BFGS-B with soft-min surrogate.
    This can escape local optima of the ring parameterization.
    """
    if time.time() > deadline - 3.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Use soft-min for smooth gradient, then verify with exact min
    def neg_soft_score(x):
        pts = x.reshape(-1, 2)
        return -_soft_min_score(pts, beta=50.0)
    
    try:
        res = minimize(
            neg_soft_score,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        polished = res.x.reshape(-1, 2)
        polished_score = _calculate_score(polished)
        
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
    except Exception:
        pass
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    best_score = 0.0
    best_points = None
    
    # 1. Try Regular 14-gon explicitly
    # This corresponds to r1=1, theta1=0, r2=1, theta2=pi/7 (or similar)
    # Actually, regular 14-gon is just one ring of 14 points, which doesn't fit the 2-ring model directly
    # But we can approximate it with r1=1, theta1=0, r2=1, theta2=pi/7
    reg_params = [1.0, 0.0, 1.0, np.pi/7]
    reg_points = _get_ring_points(*reg_params)
    reg_score = _calculate_score(reg_points)
    best_score = reg_score
    best_points = reg_points
    
    # 2. Run Differential Evolution on 4 parameters
    # Bounds:
    # r1, r2: [0.1, 2.0]
    # theta1: [0, 2*pi/7]
    # theta2: [0, 2*pi/7]
    bounds = [
        (0.1, 2.0),  # r1
        (0.0, 2 * np.pi / 7),  # theta1
        (0.1, 2.0),  # r2
        (0.0, 2 * np.pi / 7)   # theta2
    ]
    
    # Use DE to find good ring parameters
    if time.time() < deadline - 30:  # Leave time for polish
        try:
            result = differential_evolution(
                _score_ring_params,
                bounds,
                maxiter=100,
                popsize=15,
                tol=0.001,
                seed=42,
                polish=True
            )
            
            if result.fun < 0:
                de_score = -result.fun
                if de_score > best_score:
                    best_score = de_score
                    best_points = _get_ring_points(*result.x)
        except Exception:
            pass
            
    # 3. Coordinate-level polish using soft-min surrogate
    # This allows breaking the 7-fold symmetry if beneficial
    if time.time() < deadline - 10 and best_points is not None:
        polished_points, polished_score = _coordinate_polish(best_points, 10.0, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # 4. Final check and return
    if best_points is None:
        # Fallback to random if something went wrong
        rng = np.random.RandomState(N)
        best_points = rng.rand(N, 2)
        
    return best_points
# EVOLVE-BLOCK-END
