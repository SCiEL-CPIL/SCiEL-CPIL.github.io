import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area_vec(points):
    """Exact minimum triangle area over all C(n,3) triples."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    # Use a loop-based approach to avoid the large memory allocation of np.indices
    # For n=14, C(14,3) = 364, so a simple loop is fine and memory efficient
    m = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j,0] - points[i,0]) * (points[k,1] - points[i,1]) -
                                 (points[j,1] - points[i,1]) * (points[k,0] - points[i,0]))
                if area < m:
                    m = area
    return m

def _hull_area_fast(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0

def _score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    ha = _hull_area_fast(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area_vec(points)
    if ma <= 0.0:
        return 0.0
    return ma / ha

def _regular_polygon(k, r=1.0, phase=0.0):
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.column_stack([np.cos(a), np.sin(a)])

def _normalize_unit_area(points):
    """Scale so convex hull area = 1."""
    try:
        ha = ConvexHull(points).volume
        if ha > 1e-10:
            return points * np.sqrt(1.0 / ha)
    except Exception:
        pass
    xs, ys = points[:, 0], points[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        return points / np.sqrt(w * h)
    return points

def _objective_free_points(params):
    """
    Objective function for free point optimization.
    params: 2N values, [x1, y1, ..., xN, yN]
    We maximize min_triangle_area / hull_area.
    To use minimization, we return -ratio.
    """
    if np.any(~np.isfinite(params)):
        return 1e10
    
    pts = params.reshape(N, 2)
    
    # Check for degeneracy
    if np.any(np.abs(pts) > 100):
        return 1e10
        
    ha = _hull_area_fast(pts)
    if ha < 1e-12:
        return 1e10
        
    ma = _min_triangle_area_vec(pts)
    if ma <= 0:
        return 1e10
        
    ratio = ma / ha
    return -ratio

def _construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    best_score = 0.0
    best_pts = None
    
    # Initial best: Regular 14-gon
    reg_pts = _regular_polygon(N)
    best_score = _score(reg_pts)
    best_pts = reg_pts.copy()
    
    # Strategy:
    # 1. Start with regular polygon.
    # 2. Optimize free coordinates using Powell.
    # 3. Use multiple restarts with different initial perturbations.
    
    rng = np.random.default_rng(12345)
    
    # Generate initial candidates
    starts = []
    
    # 1. Regular polygon
    starts.append(reg_pts.flatten())
    
    # 2. Slightly perturbed regular polygon
    for i in range(20):
        if time.time() > deadline - 30:
            break
        noise = rng.normal(0, 0.1, (N, 2))
        pts = reg_pts + noise
        starts.append(pts.flatten())
        
    # 3. Random convex sets? Hard to generate. Let's stick to perturbations of regular.
    # Maybe try different scales or rotations? Regular is scale/rotation invariant in ratio.
    # So perturbations are the key.
    
    # Add some more aggressive perturbations
    for i in range(10):
        if time.time() > deadline - 30:
            break
        noise = rng.normal(0, 0.3, (N, 2))
        pts = reg_pts + noise
        starts.append(pts.flatten())

    # Optimize
    for idx, start in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        try:
            # Use Powell method which doesn't require gradients
            # maxiter needs to be sufficient but not too slow
            # For 28 variables, Powell can be slow.
            # Let's try Nelder-Mead or COBYLA?
            # Powell is robust.
            
            # Limit iterations to keep time down
            max_iter = 200 if len(starts) > 10 else 500
            
            res = minimize(_objective_free_points, start, method='Powell',
                           options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-8})
            
            if np.isfinite(res.fun):
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
                    
        except Exception:
            pass

    # If we have time, try a second round of optimization from the best found
    if time.time() < deadline - 10 and best_pts is not None:
        try:
            start = best_pts.flatten()
            res = minimize(_objective_free_points, start, method='Powell',
                           options={'maxiter': 300, 'xtol': 1e-7, 'ftol': 1e-9})
            if np.isfinite(res.fun):
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    return _normalize_unit_area(best_pts)

def construct():
    return _construct()
