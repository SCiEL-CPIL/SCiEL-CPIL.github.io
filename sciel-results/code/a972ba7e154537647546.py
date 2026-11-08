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

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _make_points_from_params(params):
    """
    Generate 13 points from a multi-shell parameterization.
    params: [r1, theta1, r2, theta2, r3, theta3]
    where we have 3 shells with 5, 5, and 3 points respectively.
    r_i: radius of shell i
    theta_i: rotation phase of shell i
    """
    r1, theta1, r2, theta2, r3, theta3 = params
    
    # Shell 1: 5 points (regular pentagon)
    angles1 = theta1 + 2 * np.pi * np.arange(5) / 5
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    
    # Shell 2: 5 points (regular pentagon)
    angles2 = theta2 + 2 * np.pi * np.arange(5) / 5
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    
    # Shell 3: 3 points (equilateral triangle)
    angles3 = theta3 + 2 * np.pi * np.arange(3) / 3
    pts3 = np.column_stack((r3 * np.cos(angles3), r3 * np.sin(angles3)))
    
    points = np.vstack([pts1, pts2, pts3])
    return points

def _objective(params, T=100.0):
    """
    Objective to minimize: -log(min_triangle_area) + log(hull_area)
    Using smooth min approximation.
    """
    points = _make_points_from_params(params)
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1e10
    
    areas = _all_triangle_areas(points)
    if np.any(areas <= 1e-15):
        return 1e10
    
    # Smooth min: min(a) ≈ -(1/T) * log(sum(exp(-T*a)))
    # We want to minimize: -log(min_area) + log(hull_area)
    neg_T_areas = -T * areas
    max_val = np.max(neg_T_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
    
    obj = log_sum_exp + np.log(T) + np.log(hull_area)
    return obj

def _objective_raw(params):
    """
    Direct objective: minimize -score (i.e., maximize min_area/hull_area)
    """
    points = _make_points_from_params(params)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Try multiple multi-shell configurations
    # Configuration 1: 5-5-3 shells
    # Initial guess: concentric shells with optimal radii
    initial_params_list = [
        # [r1, theta1, r2, theta2, r3, theta3]
        [0.3, 0.0, 0.7, 0.0, 1.0, 0.0],
        [0.2, 0.0, 0.6, 0.3, 1.0, 0.0],
        [0.4, 0.1, 0.8, 0.2, 1.0, 0.1],
        [0.3, 0.0, 0.7, 0.628, 1.0, 0.0],  # 36 degree offset
        [0.25, 0.0, 0.65, 0.628, 1.0, 0.0],
        [0.35, 0.0, 0.75, 0.628, 1.0, 0.0],
        [0.3, 0.3, 0.7, 0.9, 1.0, 0.5],
        [0.2, 0.0, 0.5, 0.0, 0.9, 0.0],
        [0.4, 0.0, 0.8, 0.0, 1.0, 0.0],
    ]
    
    for init_params in initial_params_list:
        if time.time() > deadline - 10.0:
            break
        
        x0 = np.array(init_params, dtype=float)
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0.01, 1.5), (-np.pi, np.pi), (0.01, 1.5), (-np.pi, np.pi), (0.01, 1.5), (-np.pi, np.pi)],
                options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5.0:
                break
            
            candidate_params = result.x
            candidate_points = _make_points_from_params(candidate_params)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        except Exception:
            continue
    
    # If multi-shell didn't work well, try a different approach:
    # 7-6 split or other configurations
    
    # Try 6-7 split
    def _make_points_6_7(params):
        """
        Generate 13 points: 6 on inner hexagon, 7 on outer heptagon
        params: [r1, theta1, r2, theta2]
        """
        r1, theta1, r2, theta2 = params
        
        angles1 = theta1 + 2 * np.pi * np.arange(6) / 6
        pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        
        angles2 = theta2 + 2 * np.pi * np.arange(7) / 7
        pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        
        return np.vstack([pts1, pts2])
    
    def _objective_6_7(params, T=100.0):
        points = _make_points_6_7(params)
        hull_area = _convex_hull_area(points)
        if hull_area <= 1e-12:
            return 1e10
        areas = _all_triangle_areas(points)
        if np.any(areas <= 1e-15):
            return 1e10
        neg_T_areas = -T * areas
        max_val = np.max(neg_T_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
        return log_sum_exp + np.log(T) + np.log(hull_area)
    
    init_params_6_7 = [
        [0.4, 0.0, 1.0, 0.0],
        [0.3, 0.0, 1.0, 0.3],
        [0.5, 0.0, 1.0, 0.0],
        [0.4, 0.0, 1.0, 0.45],
        [0.35, 0.0, 1.0, 0.45],
    ]
    
    for init_params in init_params_6_7:
        if time.time() > deadline - 10.0:
            break
        
        x0 = np.array(init_params, dtype=float)
        
        try:
            result = minimize(
                _objective_6_7,
                x0,
                method='L-BFGS-B',
                bounds=[(0.01, 1.5), (-np.pi, np.pi), (0.01, 1.5), (-np.pi, np.pi)],
                options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5.0:
                break
            
            candidate_params = result.x
            candidate_points = _make_points_6_7(candidate_params)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        except Exception:
            continue
    
    # If best_points is still None, fall back to regular 13-gon
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
    
    # Try to improve with random restarts around the best solution
    if best_points is not None and time.time() < deadline - 15.0:
        rng = np.random.RandomState(42)
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb the best points slightly
            perturbation = rng.randn(N, 2) * 0.05
            candidate_points = best_points + perturbation
            
            # Try to optimize from this perturbed configuration
            x0 = candidate_points.flatten()
            
            def _obj_coords(x):
                pts = x.reshape(-1, 2)
                hull_area = _convex_hull_area(pts)
                if hull_area <= 1e-12:
                    return 1e10
                areas = _all_triangle_areas(pts)
                if np.any(areas <= 1e-15):
                    return 1e10
                neg_T_areas = -50.0 * areas
                max_val = np.max(neg_T_areas)
                log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
                return log_sum_exp + np.log(50.0) + np.log(hull_area)
            
            try:
                result = minimize(
                    _obj_coords,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if time.time() > deadline - 5.0:
                    break
                
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
            except Exception:
                continue
    
    return best_points
# EVOLVE-BLOCK-END
