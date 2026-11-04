import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _triangle_areas(points):
    n = len(points)
    areas = np.zeros((n, n, n)) # This is too big. 14^3 = 2744. It's fine.
    # Actually, we only need i<j<k.
    # Let's just compute a flat array of areas for i<j<k
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except:
        return 0.0

def _exact_score(points):
    areas = _triangle_areas(points)
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    return np.min(areas) / hull_a

def _surrogate_score(points, alpha=10.0):
    areas = _triangle_areas(points)
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    # Soft min
    # To avoid underflow/overflow, shift areas
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum_exp = min_a + np.log(np.sum(np.exp(-alpha * shifted)))
    soft_min = -log_sum_exp # This is the soft min value
    # We want to maximize soft_min / hull_a
    # So minimize -soft_min / hull_a
    return -soft_min / hull_a

def _normalize(points):
    # Center
    c = np.mean(points, axis=0)
    points = points - c
    # Scale to unit variance (sum of squared norms = 1)
    norm = np.sqrt(np.sum(points**2))
    if norm < 1e-10:
        return points
    points = points / norm
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial guess: Jittered 3x5 grid
    # 3 rows, 5 cols
    # Remove one point to get 14
    grid_pts = []
    for i in range(3):
        for j in range(5):
            grid_pts.append([i/2.0, j/4.0])
    grid_pts = np.array(grid_pts[:14]) # Take first 14
    # Add jitter
    np.random.seed(42)
    grid_pts += 0.1 * np.random.randn(14, 2)
    
    best_points = _normalize(grid_pts)
    best_score = _exact_score(best_points)
    
    # Optimization 1: L-BFGS-B on surrogate
    try:
        res = minimize(
            _surrogate_score, 
            best_points.flatten(), 
            args=(10.0,), 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        if time.time() < deadline:
            candidate = _normalize(res.x.reshape(N, 2))
            score = _exact_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate
    except Exception as e:
        pass
        
    # Optimization 2: Nelder-Mead on exact score
    if time.time() < deadline - 10.0: # Need 10s for NM
        try:
            res = minimize(
                lambda x: -_exact_score(_normalize(x.reshape(N, 2))), 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if time.time() < deadline:
                candidate = _normalize(res.x.reshape(N, 2))
                score = _exact_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
        except Exception as e:
            pass

    # Try a different start if time permits
    if time.time() < deadline - 20.0:
        # Random start
        np.random.seed(123)
        start_pts = np.random.randn(14, 2)
        start_pts = _normalize(start_pts)
        
        try:
            res = minimize(
                _surrogate_score, 
                start_pts.flatten(), 
                args=(10.0,), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if time.time() < deadline - 10.0:
                candidate = _normalize(res.x.reshape(N, 2))
                score = _exact_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
        except Exception as e:
            pass
        
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    lambda x: -_exact_score(_normalize(x.reshape(N, 2))), 
                    best_points.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if time.time() < deadline:
                    candidate = _normalize(res.x.reshape(N, 2))
                    score = _exact_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate
            except Exception as e:
                pass

    return best_points
