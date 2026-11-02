import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 0.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _decode_convex_params(params):
    """
    params: [angle_offsets (N), radius_offsets (N), aspect_ratio]
    Returns points in convex position.
    """
    N = len(params) - 1
    angle_offsets = params[:N]
    radius_offsets = params[N:-1]
    aspect_ratio = params[-1]
    
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Sort angles to maintain convexity (cyclic order)
    angles = np.sort(base_angles + angle_offsets)
    
    # Ensure angles are spread out enough to avoid degenerate triangles?
    # Actually, sorting handles the order. We just need to ensure no overlap.
    # Since offsets are small, this should be fine.
    
    radii = 1.0 + radius_offsets
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles) * aspect_ratio
    
    return np.column_stack((x, y))

def _objective(params):
    points = _decode_convex_params(params)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial guess: Regular polygon
    # params: [0]*N + [0]*N + [1.0]
    x0 = np.zeros(N + N + 1)
    x0[-1] = 1.0
    
    best_params = x0.copy()
    best_score = _score(_decode_convex_params(best_params))
    best_points = _decode_convex_params(best_params)
    
    # Multi-start local search
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        if i == 0:
            current_params = x0.copy()
        else:
            # Random perturbation
            current_params = x0.copy()
            current_params[:-1] += np.random.uniform(-0.1, 0.1, size=2*N)
            current_params[-1] = 1.0 + np.random.uniform(-0.2, 0.2)
        
        try:
            res = minimize(
                _objective,
                current_params,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'maxiter': 500,
                    'adaptive': True
                }
            )
            
            curr_score = -res.fun
            if curr_score > best_score:
                best_score = curr_score
                best_params = res.x
                best_points = _decode_convex_params(res.x)
        except Exception:
            pass
        
        # Also try Powell for a few starts
        if i < 5 and time.time() < deadline - 10:
            try:
                res_powell = minimize(
                    _objective,
                    current_params,
                    method='Powell',
                    options={
                        'xtol': 1e-6,
                        'ftol': 1e-6,
                        'maxiter': 100
                    }
                )
                curr_score = -res_powell.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_params = res_powell.x
                    best_points = _decode_convex_params(res_powell.x)
            except Exception:
                pass

    # Final refinement with high precision on the best found
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'maxiter': 1000,
                    'adaptive': True
                }
            )
            curr_score = -res_final.fun
            if curr_score > best_score:
                best_score = curr_score
                best_params = res_final.x
                best_points = _decode_convex_params(res_final.x)
        except Exception:
            pass

    return best_points
