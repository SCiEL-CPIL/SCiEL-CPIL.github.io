import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    n = points.shape[0]
    
    # Calculate minimum triangle area
    min_area = np.inf
    
    # Loop over all triples
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Cross product magnitude gives 2 * area
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 0:
        return 0.0

    return min_area / hull_area

def _negative_score(flat_points):
    points = flat_points.reshape(N, 2)
    return -_calculate_score(points)

def construct():
    start_time = time.time()
    timeout = 135.0  # Safety margin for 150s limit
    
    # 1. Initialize with Regular Polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = _calculate_score(init_points)
    best_points = init_points.copy()
    
    # 2. Global Optimization using Differential Evolution
    # This explores the landscape more broadly than Powell starting from a single point.
    # We define bounds. Since the problem is scale-invariant (score is a ratio),
    # we can fix the scale, e.g., constrain points to be within [-1, 1]^2 or similar.
    # However, DE works best with defined bounds. Let's use [-2, 2] for x and y.
    bounds = [(-2, 2)] * (2 * N)
    
    def de_callback(xk, convergence):
        nonlocal best_score, best_points
        nonlocal start_time, timeout
        if time.time() - start_time > timeout:
            raise RuntimeError("Timeout")
        # Evaluate current best from DE
        pts = xk.reshape(N, 2)
        s = _calculate_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    try:
        res_de = differential_evolution(
            _negative_score, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8, 
            polish=False, # We will polish manually to save time and control
            callback=de_callback,
            popsize=15
        )
    except RuntimeError:
        pass
    except Exception:
        pass

    # 3. Local Refinement
    # If we have time, polish the best candidate found so far.
    # We use Powell for its robustness on non-smooth functions.
    if time.time() - start_time < timeout - 20:
        try:
            res_powell = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Powell', 
                options={'maxiter': 500, 'disp': False, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            if res_powell.x is not None:
                candidate_points = res_powell.x.reshape(N, 2)
                cand_score = _calculate_score(candidate_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate_points
        except Exception:
            pass

    # 4. Further Refinement with Nelder-Mead if time permits
    if time.time() - start_time < timeout - 10:
        try:
            res_nm = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'disp': False, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_nm.x is not None:
                candidate_points2 = res_nm.x.reshape(N, 2)
                cand_score2 = _calculate_score(candidate_points2)
                if cand_score2 > best_score:
                    best_score = cand_score2
                    best_points = candidate_points2
        except Exception:
            pass

    return best_points
