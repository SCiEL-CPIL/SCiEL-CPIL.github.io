import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, LinearConstraint

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
        if cross <= 1e-9:
            return False
    return True

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    dists_to_centroid = np.linalg.norm(polygon_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return None
    return polygon_norm / mean_dist

def _soft_min(values, temperature):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    min_val = np.min(values)
    if temperature <= 0:
        return min_val
    shifted = values - min_val
    exp_vals = np.exp(-shifted / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return min_val - temperature * log_sum

def _soft_total_score(polygon, temperature):
    """Calculate soft minimum vertex score over all vertices."""
    scores = np.array([_vertex_score(polygon, i) for i in range(polygon.shape[0])])
    return _soft_min(scores, temperature)

def _support_function_to_polygon(h_coeffs, n, angles):
    """
    Convert Fourier coefficients of support function to polygon vertices.
    h(θ) = c0 + sum_m (a_m * cos(m*θ) + b_m * sin(m*θ))
    
    The polygon is the intersection of half-planes: x*cos(θ_i) + y*sin(θ_i) <= h(θ_i)
    We find vertices as intersections of adjacent half-planes.
    """
    n_harmonics = (len(h_coeffs) - 1) // 2
    R = np.zeros(n)
    R += h_coeffs[0]
    for m in range(1, n_harmonics + 1):
        idx_c = 2 * m - 1
        idx_s = 2 * m
        if idx_c >= len(h_coeffs) or idx_s >= len(h_coeffs):
            break
        R += h_coeffs[idx_c] * np.cos(m * angles)
        R += h_coeffs[idx_s] * np.sin(m * angles)
    
    # R[i] = h(θ_i) is the support value at angle θ_i
    # Half-plane: x*cos(θ_i) + y*sin(θ_i) <= R[i]
    # Vertex i is intersection of half-planes i-1 and i
    # Solve: [cos(θ_{i-1}), sin(θ_{i-1})] [x,y]^T = R[i-1]
    #        [cos(θ_i),     sin(θ_i)]     [x,y]^T = R[i]
    
    vertices = np.zeros((n, 2))
    for i in range(n):
        i_prev = (i - 1) % n
        theta_prev = angles[i_prev]
        theta_curr = angles[i]
        
        a1 = np.cos(theta_prev)
        b1 = np.sin(theta_prev)
        c1 = R[i_prev]
        
        a2 = np.cos(theta_curr)
        b2 = np.sin(theta_curr)
        c2 = R[i]
        
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            # Degenerate case, use midpoint
            vertices[i] = [(c1 + c2) / 2, 0]
        else:
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            vertices[i] = [x, y]
    
    return vertices

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    R_reg = np.ones(n)
    polygon_reg = _support_function_to_polygon(np.array([1.0] + [0.0] * 8), n, angles)
    polygon_reg = _normalize(polygon_reg)
    if polygon_reg is None:
        # Fallback: return regular polygon
        x = R_reg * np.cos(angles)
        y = R_reg * np.sin(angles)
        return np.stack([x, y], axis=1)
        
    score_reg = _total_score(polygon_reg)
    best_polygon = polygon_reg.copy()
    current_best = score_reg
    
    start_time = time.time()
    time_limit = 600.0  # Safety margin before 630s kill
    
    # Fourier representation of support function: DC + up to 5 harmonics = 11 parameters
    n_harmonics = 7
    n_params = 1 + 2 * n_harmonics
    
    def get_polygon_from_h_coeffs(h_coeffs):
        polygon = _support_function_to_polygon(h_coeffs, n, angles)
        # Check for finite values
        if not np.all(np.isfinite(polygon)):
            return None
        # Check no repeated points
        for i in range(n):
            if np.linalg.norm(polygon[i] - polygon[(i+1) % n]) < 1e-10:
                return None
        return polygon
    
    def objective_h(h_coeffs, temperature=0.05):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            return 1e9
            
        polygon = get_polygon_from_h_coeffs(h_coeffs)
        if polygon is None:
            return 1e9
            
        if not _is_convex(polygon):
            return 1e9
            
        s = _soft_total_score(polygon, temperature)
        return -s
        
    # Phase 1: Differential Evolution for global search
    # Bounds: DC component around 1, harmonics small with decay
    bounds = [(0.8, 1.2)]  # DC
    for m in range(1, n_harmonics + 1):
        amp = 0.12 / m
        bounds.append((-amp, amp))
        bounds.append((-amp, amp))
    
    # Multiple restarts with different seeds
    n_restarts = 8
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 120:
            break
            
        seed = 42 + restart * 137
        
        # Perturb bounds for diversity
        if restart == 0:
            bounds_cur = bounds
        else:
            bounds_cur = []
            for b in bounds:
                center = (b[0] + b[1]) / 2
                half = (b[1] - b[0]) / 2
                offset = np.random.uniform(-0.02, 0.02)
                bounds_cur.append((center - half + offset, center + half + offset))
        
        try:
            res = differential_evolution(
                lambda x: objective_h(x, temperature=0.1),
                bounds=bounds_cur,
                seed=seed,
                maxiter=200,
                tol=1e-7,
                popsize=12,
                mutation=(0.6, 1.2),
                recombination=0.7,
                polish=False
            )
            
            if res.fun < -current_best:
                h_coeffs_opt = res.x
                polygon_opt = get_polygon_from_h_coeffs(h_coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    polygon_norm = _normalize(polygon_opt)
                    if polygon_norm is not None:
                        score_opt = _total_score(polygon_norm)
                        if score_opt > current_best:
                            best_polygon = polygon_norm
                            current_best = score_opt
        except Exception:
            pass
    
    # Phase 2: Local refinement with annealing temperature
    elapsed = time.time() - start_time
    if elapsed < time_limit - 90:
        try:
            # Extract initial h coefficients from best polygon
            # For a polygon with vertices v_i, h(θ_i) = v_i · [cos(θ_i), sin(θ_i)]
            h_init = np.zeros(n_params)
            h_init[0] = np.mean(np.sum(best_polygon * np.column_stack([np.cos(angles), np.sin(angles)]), axis=1))
            for m in range(1, n_harmonics + 1):
                idx_c = 2 * m - 1
                idx_s = 2 * m
                if idx_c >= n_params or idx_s >= n_params:
                    break
                h_vals = np.sum(best_polygon * np.column_stack([np.cos(angles), np.sin(angles)]), axis=1)
                h_init[idx_c] = np.mean(h_vals * np.cos(m * angles))
                h_init[idx_s] = np.mean(h_vals * np.sin(m * angles))
            
            # Annealing: start with higher temperature, decrease
            temperatures = [0.1, 0.05, 0.02, 0.01]
            for temp in temperatures:
                elapsed = time.time() - start_time
                if elapsed > time_limit - 30:
                    break
                    
                def objective_local(h_coeffs):
                    elapsed_local = time.time() - start_time
                    if elapsed_local > time_limit - 15:
                        return 1e9
                        
                    polygon = get_polygon_from_h_coeffs(h_coeffs)
                    if polygon is None:
                        return 1e9
                        
                    if not _is_convex(polygon):
                        return 1e9
                        
                    s = _soft_total_score(polygon, temp)
                    return -s
                    
                res_local = minimize(objective_local, h_init, method='Nelder-Mead',
                                     options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res_local.fun < -current_best:
                    h_coeffs_opt = res_local.x
                    polygon_opt = get_polygon_from_h_coeffs(h_coeffs_opt)
                    
                    if polygon_opt is not None and _is_convex(polygon_opt):
                        polygon_norm = _normalize(polygon_opt)
                        if polygon_norm is not None:
                            score_opt = _total_score(polygon_norm)
                            if score_opt > current_best:
                                best_polygon = polygon_norm
                                current_best = score_opt
                                # Update h_init for next temperature
                                h_vals = np.sum(best_polygon * np.column_stack([np.cos(angles), np.sin(angles)]), axis=1)
                                h_init = np.zeros(n_params)
                                h_init[0] = np.mean(h_vals)
                                for m in range(1, n_harmonics + 1):
                                    idx_c = 2 * m - 1
                                    idx_s = 2 * m
                                    if idx_c >= n_params or idx_s >= n_params:
                                        break
                                    h_init[idx_c] = np.mean(h_vals * np.cos(m * angles))
                                    h_init[idx_s] = np.mean(h_vals * np.sin(m * angles))
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with exact objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            h_vals = np.sum(best_polygon * np.column_stack([np.cos(angles), np.sin(angles)]), axis=1)
            h_init = np.zeros(n_params)
            h_init[0] = np.mean(h_vals)
            for m in range(1, n_harmonics + 1):
                idx_c = 2 * m - 1
                idx_s = 2 * m
                if idx_c >= n_params or idx_s >= n_params:
                    break
                h_init[idx_c] = np.mean(h_vals * np.cos(m * angles))
                h_init[idx_s] = np.mean(h_vals * np.sin(m * angles))
            
            def objective_fine(h_coeffs):
                elapsed_fine = time.time() - start_time
                if elapsed_fine > time_limit - 5:
                    return 1e9
                    
                polygon = get_polygon_from_h_coeffs(h_coeffs)
                if polygon is None:
                    return 1e9
                    
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            res_fine = minimize(objective_fine, h_init, method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res_fine.fun < -current_best:
                h_coeffs_opt = res_fine.x
                polygon_opt = get_polygon_from_h_coeffs(h_coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    polygon_norm = _normalize(polygon_opt)
                    if polygon_norm is not None:
                        score_opt = _total_score(polygon_norm)
                        if score_opt > current_best:
                            best_polygon = polygon_norm
                            current_best = score_opt
        except Exception:
            pass
    
    # Phase 4: Direct vertex optimization with convexity constraint
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20:
        try:
            # Work directly with vertex coordinates
            # Flatten polygon to 40-d vector
            x0 = best_polygon.flatten().copy()
            
            def objective_vertices(x):
                elapsed_v = time.time() - start_time
                if elapsed_v > time_limit - 5:
                    return 1e9
                    
                poly = x.reshape(n, 2)
                if not np.all(np.isfinite(poly)):
                    return 1e9
                    
                # Check convexity
                if not _is_convex(poly):
                    return 1e9
                    
                # Normalize
                poly_norm = _normalize(poly)
                if poly_norm is None:
                    return 1e9
                    
                s = _total_score(poly_norm)
                return -s
            
            res_v = minimize(objective_vertices, x0, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res_v.fun < -current_best:
                poly_v = res_v.x.reshape(n, 2)
                if _is_convex(poly_v):
                    poly_v_norm = _normalize(poly_v)
                    if poly_v_norm is not None:
                        score_v = _total_score(poly_v_norm)
                        if score_v > current_best:
                            best_polygon = poly_v_norm
                            current_best = score_v
        except Exception:
            pass
    
    # Final validation and return
    if _is_convex(best_polygon):
        # Ensure proper normalization
        final_norm = _normalize(best_polygon)
        if final_norm is not None and _is_convex(final_norm):
            return final_norm
        else:
            return best_polygon
    else:
        # Fallback to regular polygon
        x = R_reg * np.cos(angles)
        y = R_reg * np.sin(angles)
        return np.stack([x, y], axis=1)
