import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _triangle_areas(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    return _triangle_areas(points).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calc_ratio(points):
    try:
        if len(points) < 3:
            return 0.0
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_rings(r1, r2, r3, a1, a2, a3, n1=6, n2=6, n3=2):
    """
    Generate points on 3 concentric rings.
    Ring 1: n1 points at radius r1, angular offset a1
    Ring 2: n2 points at radius r2, angular offset a2
    Ring 3: n3 points at radius r3, angular offset a3
    """
    pts = []
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + a1
    pts.extend(np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)]))
    
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + a2
    pts.extend(np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)]))
    
    angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + a3
    pts.extend(np.column_stack([r3 * np.cos(angles3), r3 * np.sin(angles3)]))
    
    return np.array(pts)

def _normalize_points(points):
    try:
        area = _hull_area(points)
        if area < 1e-10:
            return points
        scale_factor = 1.0 / np.sqrt(area)
        return points * scale_factor
    except Exception:
        return points

def _soft_min(values, tau):
    values = np.asarray(values)
    v_max = values.max()
    if v_max < 1e-15:
        return 0.0
    scaled = -(values - v_max) / tau
    scaled = np.clip(scaled, -500, 500)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return v_max - tau * log_sum

def _neg_ring_score(params, tau=0.05):
    """
    Objective for ring parameter optimization.
    params: [r1, r2, r3, a1, a2, a3]
    """
    try:
        r1, r2, r3, a1, a2, a3 = params
        # Ensure radii are positive and ordered
        r1 = max(r1, 0.1)
        r2 = max(r2, r1 + 0.1)
        r3 = max(r3, r2 + 0.1)
        
        points = _concentric_rings(r1, r2, r3, a1, a2, a3)
        points = _normalize_points(points)
        
        areas = _triangle_areas(points)
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 1.0
        
        soft_min_val = _soft_min(areas, tau)
        score = soft_min_val / hull_area
        return -score
    except Exception:
        return 1.0

def _neg_exact_ring_score(params):
    try:
        r1, r2, r3, a1, a2, a3 = params
        r1 = max(r1, 0.1)
        r2 = max(r2, r1 + 0.1)
        r3 = max(r3, r2 + 0.1)
        
        points = _concentric_rings(r1, r2, r3, a1, a2, a3)
        points = _normalize_points(points)
        score = calc_ratio(points)
        return -score
    except Exception:
        return 1.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Optimize 3-ring configuration (6 parameters)
    # This is a much lower-dimensional problem than 28D
    initial_params = [0.3, 0.7, 1.0, 0.0, np.pi/6, np.pi/3]
    
    # Try multiple starting points for the ring optimization
    initial_guesses = [
        [0.3, 0.7, 1.0, 0.0, np.pi/6, np.pi/3],
        [0.2, 0.6, 1.0, 0.0, 0.0, 0.0],
        [0.4, 0.8, 1.2, np.pi/12, np.pi/6, np.pi/4],
        [0.25, 0.65, 1.0, np.pi/18, np.pi/9, np.pi/6],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
            
        # Anneal tau for better convergence
        tau_schedule = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        current_params = guess.copy()
        
        for tau in tau_schedule:
            if time.time() > deadline - 15:
                break
                
            try:
                res = scipy.optimize.minimize(
                    lambda p, t=tau: _neg_ring_score(p, t),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res.fun < 0:  # Valid score
                    current_params = res.x
                    # Evaluate exact score
                    r1, r2, r3, a1, a2, a3 = current_params
                    r1 = max(r1, 0.1)
                    r2 = max(r2, r1 + 0.1)
                    r3 = max(r3, r2 + 0.1)
                    points = _concentric_rings(r1, r2, r3, a1, a2, a3)
                    points = _normalize_points(points)
                    score = calc_ratio(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        
            except Exception:
                pass
    
    # Strategy 2: If we have a good starting point, do full 28D optimization
    if best_points is not None and time.time() < deadline - 20:
        # Use the best ring configuration as initial guess for full optimization
        current_points = best_points.copy()
        
        # Soft-min annealing for full 28D
        tau_schedule = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for tau in tau_schedule:
            if time.time() > deadline - 10:
                break
                
            try:
                def _neg_smooth_full(x, t=tau):
                    points = x.reshape(-1, 2)
                    try:
                        areas = _triangle_areas(points)
                        hull_area = _hull_area(points)
                        if hull_area < 1e-10:
                            return 1.0
                        soft_min_val = _soft_min(areas, t)
                        score = soft_min_val / hull_area
                        return -score
                    except Exception:
                        return 1.0
                
                res = scipy.optimize.minimize(
                    _neg_smooth_full,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                optimized_points = res.x.reshape(-1, 2)
                optimized_points = _normalize_points(optimized_points)
                score = calc_ratio(optimized_points)
                
                if score > best_score:
                    best_score = score
                    best_points = optimized_points.copy()
                    current_points = optimized_points
                    
            except Exception:
                pass
    
    # Strategy 3: Nelder-Mead polish on the best configuration
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_nm = scipy.optimize.minimize(
                lambda x: -calc_ratio(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            nm_points = res_nm.x.reshape(-1, 2)
            nm_points = _normalize_points(nm_points)
            score = calc_ratio(nm_points)
            if score > best_score:
                best_score = score
                best_points = nm_points
        except Exception:
            pass
    
    # Fallback: regular polygon
    if best_points is None:
        best_points = _normalize_points(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize_points(best_points)
        
    return best_points
