import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

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

def _generate_ring_points(k_inner, k_outer, r_in, r_out, alpha, beta):
    """
    Generate points for a two-ring configuration.
    """
    points = []
    
    if k_inner > 0:
        angles_in = 2 * np.pi * np.arange(k_inner) / k_inner + alpha
        x_in = r_in * np.cos(angles_in)
        y_in = r_in * np.sin(angles_in)
        points.append(np.column_stack((x_in, y_in)))
    
    if k_outer > 0:
        angles_out = 2 * np.pi * np.arange(k_outer) / k_outer + beta
        x_out = r_out * np.cos(angles_out)
        y_out = r_out * np.sin(angles_out)
        points.append(np.column_stack((x_out, y_out)))
        
    if not points:
        return np.zeros((0, 2))
        
    return np.vstack(points)

def _ring_score(params, k_inner, k_outer):
    """
    Objective function for ring optimization.
    params: [r_in, r_out, alpha, beta]
    """
    r_in, r_out, alpha, beta = params
    
    if k_inner > 1 and r_in < 1e-6:
        r_in = 1e-6
        
    if r_out < r_in:
        r_out = r_in + 1e-6

    pts = _generate_ring_points(k_inner, k_outer, r_in, r_out, alpha, beta)
    
    if pts.shape[0] != N:
        return 1.0 # Bad score (minimize -score)
        
    score = _calculate_score(pts)
    return -score

def _apply_affine_transform(points, params):
    """
    Apply an affine transformation defined by 6 parameters to the points.
    params: [a, b, c, d, e, f]
    Matrix M = [[a, b], [c, d]]
    Translation t = [e, f]
    New_points = points @ M.T + t
    """
    a, b, c, d, e, f = params
    M = np.array([[a, b], [c, d]])
    t = np.array([e, f])
    return points @ M.T + t

def _affine_score(params, base_points):
    """
    Objective for affine transformation optimization.
    """
    try:
        transformed = _apply_affine_transform(base_points, params)
        # Check for valid points
        if not np.all(np.isfinite(transformed)):
            return 100.0
        # Check for degeneracy (very small area triangles)
        score = _calculate_score(transformed)
        if score == 0.0:
            return 100.0
        return -score
    except Exception:
        return 100.0

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit
    
    best_score = -np.inf
    best_points = None
    
    # 1. Initialize with Regular Polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _calculate_score(init_points)
    best_points = init_points.copy()
    
    # 2. Optimize Ring Configurations (Current Strategy)
    # This explores the space of symmetric multi-ring solutions.
    splits = [(k, N - k) for k in range(1, N // 2 + 1)]
    
    for k_in, k_out in splits:
        if time.time() - start_time > timeout - 30:
            break
            
        def ring_opt_func(params, k_in_val, k_out_val):
            r_in, r_out, alpha, beta = params
            if r_out <= r_in:
                r_out = r_in + 1e-6
            if r_in < 0:
                r_in = 0
            if k_in_val > 1 and r_in < 1e-4:
                r_in = 1e-4
            return _ring_score([r_in, r_out, alpha, beta], k_in_val, k_out_val)

        bounds = [(0, 1.5), (1.5, 3.0), (0, 2*np.pi), (0, 2*np.pi)]
        
        try:
            res_de = differential_evolution(
                ring_opt_func, 
                bounds, 
                args=(k_in, k_out),
                seed=42 + k_in, 
                maxiter=30, 
                tol=1e-5, 
                polish=False,
                popsize=8
            )
            
            if res_de.fun < -best_score:
                r_in, r_out, alpha, beta = res_de.x
                if r_out <= r_in:
                    r_out = r_in + 1e-6
                if k_in > 1 and r_in < 1e-4:
                    r_in = 1e-4
                    
                pts = _generate_ring_points(k_in, k_out, r_in, r_out, alpha, beta)
                s = _calculate_score(pts)
                
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                    
                    # Quick local refinement
                    if time.time() - start_time < timeout - 20:
                        def local_func(p):
                            r_in_l, r_out_l, alpha_l, beta_l = p
                            if r_out_l <= r_in_l:
                                r_out_l = r_in_l + 1e-6
                            if k_in > 1 and r_in_l < 1e-4:
                                r_in_l = 1e-4
                            return _ring_score([r_in_l, r_out_l, alpha_l, beta_l], k_in, k_out)
                            
                        res_powell = minimize(
                            local_func, 
                            res_de.x, 
                            method='Powell', 
                            bounds=bounds,
                            options={'maxiter': 50, 'disp': False, 'xtol': 1e-7, 'ftol': 1e-7}
                        )
                        
                        if res_powell.fun < -best_score:
                            r_in_f, r_out_f, alpha_f, beta_f = res_powell.x
                            if r_out_f <= r_in_f:
                                r_out_f = r_in_f + 1e-6
                            if k_in > 1 and r_in_f < 1e-4:
                                r_in_f = 1e-4
                                
                            pts_f = _generate_ring_points(k_in, k_out, r_in_f, r_out_f, alpha_f, beta_f)
                            s_f = _calculate_score(pts_f)
                            
                            if s_f > best_score:
                                best_score = s_f
                                best_points = pts_f.copy()
                                
        except Exception:
            pass

    # 3. Affine Transformation Optimization (Inspiration Strategy)
    # Take the best points found so far (likely a ring config or regular polygon)
    # and try to improve them by applying a global affine transformation.
    # This allows breaking the strict circular symmetry of the ring while preserving
    # the relative angular structure, potentially finding a better non-circular hull.
    
    if best_points is not None and time.time() - start_time < timeout - 15:
        base_pts = best_points.copy()
        
        # Initial guess: Identity transformation
        # a=1, b=0, c=0, d=1, e=0, f=0
        x0 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        # Bounds for affine parameters
        # We want to allow scaling, rotation, shearing, translation.
        # Translation can be anything, but relative to the hull size it doesn't matter much for the score ratio.
        # However, to keep numbers stable, let's bound them reasonably.
        # The points are roughly in [-2, 2].
        # Scaling factors a, d can be in [0.1, 3.0].
        # Shear b, c can be in [-2.0, 2.0].
        # Translation e, f can be in [-5.0, 5.0].
        
        bounds_affine = [
            (0.1, 3.0),  # a
            (-2.0, 2.0), # b
            (-2.0, 2.0), # c
            (0.1, 3.0),  # d
            (-5.0, 5.0), # e
            (-5.0, 5.0)  # f
        ]
        
        try:
            # Use DE to find a good affine transform
            res_de_affine = differential_evolution(
                _affine_score,
                bounds_affine,
                args=(base_pts,),
                seed=123,
                maxiter=20,
                tol=1e-5,
                polish=False,
                popsize=10
            )
            
            if res_de_affine.fun < -best_score:
                params_affine = res_de_affine.x
                pts_affine = _apply_affine_transform(base_pts, params_affine)
                s_affine = _calculate_score(pts_affine)
                
                if s_affine > best_score:
                    best_score = s_affine
                    best_points = pts_affine.copy()
                    
                    # Local refinement of the affine parameters
                    if time.time() - start_time < timeout - 5:
                        res_powell_affine = minimize(
                            _affine_score,
                            params_affine,
                            method='Powell',
                            args=(base_pts,),
                            options={'maxiter': 100, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
                        )
                        
                        if res_powell_affine.fun < -best_score:
                            params_affine_ref = res_powell_affine.x
                            pts_affine_ref = _apply_affine_transform(base_pts, params_affine_ref)
                            s_affine_ref = _calculate_score(pts_affine_ref)
                            
                            if s_affine_ref > best_score:
                                best_score = s_affine_ref
                                best_points = pts_affine_ref.copy()
                                
        except Exception:
            pass

    # 4. Final Full Coordinate Polish
    # If time permits, do a quick Powell step on the full 28D coordinates of the best point found.
    # This can correct any slight asymmetries introduced by the affine transform or ring optimization.
    
    if best_points is not None and time.time() - start_time < timeout - 5:
        def full_neg_score(flat):
            pts = flat.reshape(N, 2)
            return -_calculate_score(pts)
            
        try:
            res_full = minimize(
                full_neg_score, 
                best_points.flatten(), 
                method='Powell', 
                options={'maxiter': 30, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if res_full.fun < -best_score:
                pts_full = res_full.x.reshape(N, 2)
                s_full = _calculate_score(pts_full)
                if s_full > best_score:
                    best_score = s_full
                    best_points = pts_full
        except Exception:
            pass

    return best_points
