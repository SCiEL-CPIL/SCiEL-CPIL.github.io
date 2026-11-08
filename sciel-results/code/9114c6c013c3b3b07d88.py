"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula on all points
        # (may overestimate if points are not in convex position)
        return 0.0

def _all_triangle_areas(points):
    """Compute all C(N,3) triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def _objective(x, T=50.0):
    """
    Objective to minimize: -log(min_triangle_area) + log(hull_area)
    Using log-sum-exp approximation for min.
    """
    points = x.reshape(-1, 2)
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1e10
    
    areas = _all_triangle_areas(points)
    if np.any(areas <= 1e-15):
        return 1e10
    
    # Smooth min: min(a) ≈ -(1/T) * log(sum(exp(-T*a)))
    # So -log(min(a)) ≈ log(sum(exp(-T*a))) + log(T)
    # We want to minimize: -log(min_area) + log(hull_area)
    # ≈ log(sum(exp(-T*areas))) + log(T) + log(hull_area)
    
    # For numerical stability, use the log-sum-exp trick
    neg_T_areas = -T * areas
    max_val = np.max(neg_T_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
    
    obj = log_sum_exp + np.log(T) + np.log(hull_area)
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    # Evaluate initial score
    def _score(points):
        hull_area = _convex_hull_area(points)
        if hull_area <= 1e-12:
            return 0.0
        areas = _all_triangle_areas(points)
        min_area = np.min(areas)
        return min_area / hull_area
    
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
            candidate_points = result.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    rng = np.random.RandomState(42)
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation of the regular polygon
        perturbation = rng.randn(2*N) * 0.1
        x_start = x0 + perturbation
        
        try:
            result = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5.0:
                break
            
            candidate_points = result.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
        except Exception:
            continue
    
    return best_points
# EVOLVE-BLOCK-END
