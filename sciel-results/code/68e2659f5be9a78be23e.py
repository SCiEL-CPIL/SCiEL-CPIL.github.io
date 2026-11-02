import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    # points is (N, 2)
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    # points is (N, 2)
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except:
        # Fallback if points are collinear or degenerate
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Initial guess: Regular 13-gon
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    initial_points = np.column_stack((x, y))
    initial_flat = initial_points.flatten()
    
    best_score = _score(initial_points)
    best_points = initial_points.copy()
    
    # Try local optimization
    # Nelder-Mead is good for non-smooth functions
    try:
        res = minimize(_neg_score, initial_flat, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success or res.fun < -best_score:
            candidate_points = res.x.reshape(-1, 2)
            cand_score = _score(candidate_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate_points
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline - 10:
        return best_points

    # If we have time, try a few random restarts with perturbed regular polygons
    rng = np.random.default_rng(42)
    
    for i in range(10):
        if time.time() > deadline - 5:
            break
            
        # Perturb the regular polygon
        perturbation = rng.normal(0, 0.1, N*2)
        perturbed_flat = initial_flat + perturbation
        perturbed_points = perturbed_flat.reshape(-1, 2)
        
        # Local search from perturbed start
        try:
            res = minimize(_neg_score, perturbed_flat, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_score:
                candidate_points = res.x.reshape(-1, 2)
                cand_score = _score(candidate_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate_points
        except Exception as e:
            pass

    return best_points
