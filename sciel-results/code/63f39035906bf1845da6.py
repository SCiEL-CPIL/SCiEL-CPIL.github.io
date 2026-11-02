import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area_from_polar(r, theta):
    """
    Compute min triangle area for points given in polar coordinates.
    r: array of shape (N,)
    theta: array of shape (N,)
    Returns min area.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    
    # Vectorized triangle area calculation
    # Indices for all triples i < j < k
    n = len(pts)
    min_area = float('inf')
    
    # To be fast, we can iterate i, j and vectorize k
    # But for N=14, even a simple loop is fine, but let's use numpy tricks
    # Precompute pairwise differences? 
    # Area(i,j,k) = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # Since N is small, we can just loop i, j and use numpy for k
    x = pts[:, 0]
    y = pts[:, 1]
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            # k ranges from j+1 to n-1
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_from_cartesian(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_from_polar(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    
    hull_area = _hull_area_from_cartesian(pts)
    if hull_area < 1e-10:
        return 0.0
    
    min_area = _min_triangle_area_from_polar(r, theta)
    return min_area / hull_area

def _objective_polar(params):
    """
    Objective function for minimization (minimize negative score).
    params: array of length 27: [d1, ..., d13, r1, ..., r14]
    d_i are angular gaps, r_i are radii.
    """
    d = params[:13]
    r = params[13:]
    
    # Check validity
    if not np.all(np.isfinite(params)):
        return 1e10
    
    # Ensure gaps are positive
    if np.any(d <= 0):
        return 1e10
        
    # Ensure radii are positive
    if np.any(r <= 0):
        return 1e10
        
    # Normalize angles to start at 0
    theta = np.zeros(N)
    theta[1:] = np.cumsum(d)
    # The last angle should be close to 2*pi, but we don't explicitly constrain sum(d)=2*pi in the params
    # Instead, we can just use the cumulative sum. The "shape" is determined by the relative angles.
    # However, for the hull area calculation, the absolute angle doesn't matter, only relative.
    # But wait, if we don't close the loop, is it a polygon?
    # The points are just a set. The hull is formed by these points.
    # The angular order is preserved if d_i > 0.
    
    score = _score_from_polar(r, theta)
    
    if score < 1e-10:
        return 1e10
        
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin
    
    best_score = 0.0
    best_params = None
    best_obj = float('inf')
    
    def update_best(params):
        nonlocal best_score, best_params, best_obj
        if params is None:
            return
        obj = _objective_polar(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
            best_score = -obj

    # Initial guess: Regular 14-gon
    # Angles: 2*pi/14
    # Radii: 1.0
    d_reg = np.full(13, 2 * np.pi / 14)
    r_reg = np.ones(14)
    x0 = np.concatenate([d_reg, r_reg])
    
    update_best(x0)
    
    # Phase 1: Differential Evolution
    # Bounds for d_i: [0.1, 1.5] (radians)
    # Bounds for r_i: [0.5, 2.0]
    bounds = [(0.1, 1.5)] * 13 + [(0.5, 2.0)] * 14
    
    if time.time() < deadline - 20:
        try:
            de_res = differential_evolution(
                _objective_polar, bounds,
                seed=42, maxiter=50, tol=1e-8,
                popsize=15, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x)
        except Exception:
            pass

    # Phase 2: Local Refinement with Powell
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_powell = minimize(
                _objective_polar, best_params, method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14, 'disp': False}
            )
            if res_powell.fun < best_obj:
                update_best(res_powell.x)
        except Exception:
            pass

    # Phase 3: Nelder-Mead for final polish
    if best_params is not None and time.time() < deadline - 2:
        try:
            res_nm = minimize(
                _objective_polar, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            if res_nm.fun < best_obj:
                update_best(res_nm.x)
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = x0

    # Convert best params to Cartesian points
    d = best_params[:13]
    r = best_params[13:]
    
    theta = np.zeros(N)
    theta[1:] = np.cumsum(d)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    
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
