import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

def _vertex_score(polygon, i):
    """Calculate the score for vertex i of the polygon."""
    v = polygon[i]
    diffs = polygon - v
    diffs = np.delete(diffs, i, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = np.sort(dists)
    
    if dists[0] < 0.05:
        return 0.0
        
    gaps = np.diff(dists)
    min_gap = np.min(gaps)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    if k == 0:
        ds = dists[0:4]
    elif k == 17:
        ds = dists[15:19]
    else:
        ds = dists[k-1:k+3]
        
    ratios = np.maximum(ds / D, D / ds)
    raw = np.mean(ratios)
    
    if raw == 0:
        return 1.0
    return min(1.0, 1.0 / raw)

def _total_score(polygon):
    """Calculate the minimum vertex score over all vertices."""
    min_score = 1.0
    for i in range(polygon.shape[0]):
        s = _vertex_score(polygon, i)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                return 0.0
    return min_score

def _is_convex(polygon):
    """Check if polygon is strictly convex (counter-clockwise)."""
    n = polygon.shape[0]
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 1e-12:
            return False
    return True

def _fourier_to_R(coeffs, n, angles):
    """Convert Fourier coefficients to radial values.
    coeffs[0] is the DC component (mean radius),
    coeffs[1] and coeffs[2] are cos and sin of frequency 1,
    coeffs[3] and coeffs[4] are cos and sin of frequency 2, etc.
    """
    R = np.zeros(n)
    R += coeffs[0]
    for m in range(1, len(coeffs) // 2 + 1):
        idx_c = 2 * m - 1
        idx_s = 2 * m
        if idx_c >= len(coeffs) or idx_s >= len(coeffs):
            break
        R += coeffs[idx_c] * np.cos(m * angles)
        R += coeffs[idx_s] * np.sin(m * angles)
    return R

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    R_reg = np.ones(n)
    x_reg = R_reg * np.cos(angles)
    y_reg = R_reg * np.sin(angles)
    polygon_reg = np.stack([x_reg, y_reg], axis=1)
    
    centroid = np.mean(polygon_reg, axis=0)
    polygon_reg = polygon_reg - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_reg, axis=1))
    if mean_dist > 0.01:
        polygon_reg = polygon_reg / mean_dist
    else:
        return polygon_reg
        
    score_reg = _total_score(polygon_reg)
    
    start_time = time.time()
    time_limit = 630.0
    current_best = score_reg
    best_polygon = polygon_reg.copy()
    
    def get_polygon_from_R(R):
        """Create polygon from radii at fixed angles."""
        if np.any(R <= 0.1):
            return None
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        return polygon
    
    def objective_R(R):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            return 1e9
            
        polygon = get_polygon_from_R(R)
        if polygon is None:
            return 1e9
            
        if not _is_convex(polygon):
            return 1e9 + (time.time() - start_time)
            
        s = _total_score(polygon)
        return -s
    
    # Phase 1: Differential evolution on radii (20 params)
    bounds_R = [(0.7, 1.3)] * n
    
    n_restarts = 4
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 90:
            break
            
        seed = 42 + restart * 100
        
        if restart == 0:
            bounds_cur = bounds_R
        else:
            bounds_cur = []
            for b in bounds_R:
                center = (b[0] + b[1]) / 2
                half = (b[1] - b[0]) / 2
                offset = np.random.uniform(-0.05, 0.05)
                bounds_cur.append((max(0.5, center - half + offset), center + half + offset))
        
        try:
            res = differential_evolution(
                objective_R, 
                bounds=bounds_cur,
                seed=seed,
                maxiter=100,
                tol=1e-4,
                popsize=12,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False
            )
            
            if res.fun < -current_best:
                R_opt = res.x
                polygon_opt = get_polygon_from_R(R_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    centroid_opt = np.mean(polygon_opt, axis=0)
                    polygon_opt = polygon_opt - centroid_opt
                    mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                    if mean_dist_opt > 0.01:
                        polygon_opt = polygon_opt / mean_dist_opt
                        score_opt = _total_score(polygon_opt)
                        if score_opt > current_best:
                            best_polygon = polygon_opt.copy()
                            current_best = score_opt
        except Exception:
            pass
    
    # Phase 2: Local refinement using Nelder-Mead on radii
    elapsed = time.time() - start_time
    if elapsed < time_limit - 60:
        try:
            x_best = best_polygon[:, 0]
            y_best = best_polygon[:, 1]
            R_current = np.sqrt(x_best**2 + y_best**2)
            
            def objective_local(R):
                elapsed_local = time.time() - start_time
                if elapsed_local > time_limit - 15:
                    return 1e9
                    
                polygon = get_polygon_from_R(R)
                if polygon is None:
                    return 1e9
                    
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            res_local = minimize(objective_local, R_current, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res_local.fun < -current_best:
                R_opt = res_local.x
                polygon_opt = get_polygon_from_R(R_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    centroid_opt = np.mean(polygon_opt, axis=0)
                    polygon_opt = polygon_opt - centroid_opt
                    mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                    if mean_dist_opt > 0.01:
                        polygon_opt = polygon_opt / mean_dist_opt
                        score_opt = _total_score(polygon_opt)
                        if score_opt > current_best:
                            best_polygon = polygon_opt.copy()
                            current_best = score_opt
        except Exception:
            pass
    
    # Phase 3: High-precision polish with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30 and current_best > 0:
        try:
            x_best = best_polygon[:, 0]
            y_best = best_polygon[:, 1]
            R_current = np.sqrt(x_best**2 + y_best**2)
            
            def objective_polish(R):
                elapsed_local = time.time() - start_time
                if elapsed_local > time_limit - 10:
                    return 1e9
                    
                polygon = get_polygon_from_R(R)
                if polygon is None:
                    return 1e9
                    
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            res_polish = minimize(objective_polish, R_current, method='Nelder-Mead',
                                  options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res_polish.fun < -current_best:
                R_opt = res_polish.x
                polygon_opt = get_polygon_from_R(R_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    centroid_opt = np.mean(polygon_opt, axis=0)
                    polygon_opt = polygon_opt - centroid_opt
                    mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                    if mean_dist_opt > 0.01:
                        polygon_opt = polygon_opt / mean_dist_opt
                        score_opt = _total_score(polygon_opt)
                        if score_opt > current_best:
                            best_polygon = polygon_opt.copy()
                            current_best = score_opt
        except Exception:
            pass
            
    # Final safety check
    if not _is_convex(best_polygon):
        return polygon_reg
        
    return best_polygon
