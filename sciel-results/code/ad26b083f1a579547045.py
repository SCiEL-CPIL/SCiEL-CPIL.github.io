import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    n = len(points)
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _generate_hex_model(r1, r2, theta_shift):
    """
    Generates 13 points: 1 center, 6 on circle r1, 6 on circle r2.
    The outer ring is rotated by theta_shift relative to the inner ring.
    """
    pts = np.zeros((13, 2))
    pts[0] = [0, 0]
    
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for i in range(6):
        pts[i+1] = [r1 * np.cos(angles_inner[i]), r1 * np.sin(angles_inner[i])]
        
    angles_outer = angles_inner + theta_shift
    for i in range(6):
        pts[i+7] = [r2 * np.cos(angles_outer[i]), r2 * np.sin(angles_outer[i])]
        
    return pts

def _neg_score(points):
    return -_score(points)

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 570  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            return True
        return False
    
    # 1. Initial Candidates
    # Regular 13-gon
    reg_poly = _generate_regular_polygon(13)
    update_best(reg_poly)
    
    # Hexagonal Model Candidates
    # Typical good values for hexagonal packing are r1~1, r2~sqrt(3) or 2, shift~pi/6
    initial_hex_params = [
        [1.0, 2.0, np.pi/6],
        [1.0, 1.8, np.pi/6],
        [1.0, 2.2, np.pi/6],
        [1.0, 2.0, 0.0],
        [1.0, 2.0, np.pi/3],
    ]
    
    for params in initial_hex_params:
        pts = _generate_hex_model(params[0], params[1], params[2])
        update_best(pts)
        
    if best_pts is None:
        return reg_poly
        
    # 2. Stage 1: Optimize the Hexagonal Model Parameters (3D optimization)
    # This helps find the best "shape" of the hexagonal structure before releasing all coordinates.
    
    def _hex_obj(x):
        r1, r2, theta = x
        # Keep r1, r2 positive
        r1 = abs(r1) + 0.1
        r2 = abs(r2) + 0.1
        pts = _generate_hex_model(r1, r2, theta)
        return _neg_score(pts)
        
    best_hex_params = None
    best_hex_score = 0.0
    
    for params in initial_hex_params:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(_hex_obj, params, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < -best_hex_score:
                best_hex_score = -res.fun
                best_hex_params = res.x
                # Update global best with this configuration
                r1, r2, theta = best_hex_params
                r1 = abs(r1) + 0.1
                r2 = abs(r2) + 0.1
                pts = _generate_hex_model(r1, r2, theta)
                update_best(pts)
        except Exception:
            pass
            
    # 3. Stage 2: Full 26D Nelder-Mead Optimization from Best Hex Structure
    if best_hex_params is not None:
        r1, r2, theta = best_hex_params
        r1 = abs(r1) + 0.1
        r2 = abs(r2) + 0.1
        start_pts = _generate_hex_model(r1, r2, theta).flatten()
    else:
        start_pts = best_pts.flatten()
        
    if time.time() < deadline - 60:
        try:
            res = minimize(_neg_score, start_pts, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000, 'adaptive': True})
            pts_opt = res.x.reshape(13, 2)
            update_best(pts_opt)
        except Exception:
            pass
            
    # 4. Stage 3: Full 26D Nelder-Mead Optimization from Regular Polygon
    # Sometimes the regular polygon basin is better or different.
    if time.time() < deadline - 60:
        start_pts_reg = reg_poly.flatten()
        try:
            res = minimize(_neg_score, start_pts_reg, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000, 'adaptive': True})
            pts_opt = res.x.reshape(13, 2)
            update_best(pts_opt)
        except Exception:
            pass
            
    # 5. Stage 4: Fine-tuning with smaller steps if time remains
    if time.time() < deadline - 30 and best_pts is not None:
        start_pts_fine = best_pts.flatten()
        try:
            # Tighter tolerance, fewer iterations for polish
            res = minimize(_neg_score, start_pts_fine, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
            pts_opt = res.x.reshape(13, 2)
            update_best(pts_opt)
        except Exception:
            pass
            
    if best_pts is None:
        return reg_poly
        
    return best_pts
