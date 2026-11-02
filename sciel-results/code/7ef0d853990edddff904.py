"""
Optimization approach: Direct radial optimization with fixed angular positions.
Points are placed at fixed angles (regular polygon) and their radii are optimized.
This reduces the search space from 28 to 14 dimensions (one radius per point).
We use a multi-phase strategy:
1. Powell optimization on radii (fast, direct).
2. Nelder-Mead refinement.
3. Coordinate descent for final polish.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Fixed angles for the 14 points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    def _radii_to_points(r):
        x = r * cos_a
        y = r * sin_a
        return np.column_stack([x, y])
    
    def _score_from_radii(r):
        pts = _radii_to_points(r)
        return _score(pts)
    
    def _neg_score_from_radii(r):
        r = np.maximum(r, 0.1)
        return -_score_from_radii(r)
    
    # Initial guess: Regular 14-gon (all radii = 1)
    best_r = np.ones(N)
    best_score_val = _score_from_radii(best_r)
    
    # Phase 1: Powell optimization on radii
    if time.time() < deadline - 60:
        try:
            res1 = minimize(_neg_score_from_radii, best_r, method='Powell',
                            options={'maxiter': 2000, 'ftol': 1e-12, 'xtol': 1e-12})
            r_candidate = np.maximum(res1.x, 0.1)
            score_candidate = _score_from_radii(r_candidate)
            if score_candidate > best_score_val:
                best_score_val = score_candidate
                best_r = r_candidate
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 30:
        try:
            res2 = minimize(_neg_score_from_radii, best_r, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            r_candidate = np.maximum(res2.x, 0.1)
            score_candidate = _score_from_radii(r_candidate)
            if score_candidate > best_score_val:
                best_score_val = score_candidate
                best_r = r_candidate
        except Exception:
            pass
    
    # Phase 3: Coordinate descent for final polish
    if time.time() < deadline - 5:
        for _ in range(20):  # 20 iterations of coordinate descent
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                # Optimize radius[i] while keeping others fixed
                def obj_1d(r_i):
                    r_new = best_r.copy()
                    r_new[i] = r_i
                    return -_score_from_radii(r_new)
                
                try:
                    res_cd = minimize(obj_1d, best_r[i], method='Brent',
                                      bounds=(0.1, 5.0), options={'xtol': 1e-12})
                    if res2.fun < obj_1d(best_r[i]):
                        best_r[i] = res_cd.x
                        improved = True
                except Exception:
                    pass
            
            score_candidate = _score_from_radii(best_r)
            if score_candidate > best_score_val:
                best_score_val = score_candidate
                best_r = best_r.copy()
            if not improved:
                break
    
    # Final points
    pts = _radii_to_points(best_r)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            pts = pts * scale
    except Exception:
        pass
    
    return pts
