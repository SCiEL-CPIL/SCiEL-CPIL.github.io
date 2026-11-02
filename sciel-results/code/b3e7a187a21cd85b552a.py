import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START
def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Parameterize the 14 points as (angle, radius) pairs.
    2. Sort angles to enforce a convex polygon order (points on the boundary).
    3. Use a smooth soft-min objective for the triangle areas to facilitate optimization.
    4. Run multiple diverse optimizations (Nelder-Mead, Powell) from different starts.
    5. Normalize the result to unit area.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)

    def _min_triangle_area(points):
        """Compute the minimum area of any triangle formed by 3 points."""
        min_area = float('inf')
        x = points[:, 0]
        y = points[:, 1]
        
        for i in range(N):
            for j in range(i + 1, N):
                if j + 1 >= N:
                    continue
                dx_ji = x[j] - x[i]
                dy_ji = y[j] - y[i]
                dx_ki = x[j+1:] - x[i]
                dy_ki = y[j+1:] - y[i]
                cross = dx_ji * dy_ki - dx_ki * dy_ji
                areas = 0.5 * np.abs(cross)
                if len(areas) > 0:
                    local_min = np.min(areas)
                    if local_min < min_area:
                        min_area = local_min
        return min_area

    def _score(points):
        """Compute the minimum triangle area divided by the convex hull area."""
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area < 1e-10:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / hull_area

    def _points_from_params(params):
        """
        Reconstruct points from parameters.
        params: (N, 2) array of [angle, radius]
        """
        angles = params[:, 0]
        radii = params[:, 1]
        
        # Ensure radii are positive
        radii = np.maximum(radii, 0.01)
        
        # Sort by angle to maintain convex order
        sort_idx = np.argsort(angles)
        sorted_angles = angles[sort_idx]
        sorted_radii = radii[sort_idx]
        
        x = sorted_radii * np.cos(sorted_angles)
        y = sorted_radii * np.sin(sorted_angles)
        
        return np.column_stack([x, y])

    def _objective(params_flat):
        """
        Objective function for minimization.
        We want to maximize score, so we minimize -score.
        Uses a soft-min approximation for smoothness.
        """
        params = params_flat.reshape(N, 2)
        points = _points_from_params(params)
        
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 1e10
            
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e10
            
        if hull_area < 1e-10:
            return 1e10
            
        x = points[:, 0]
        y = points[:, 1]
        
        # Compute all triangle areas
        areas = []
        for i in range(N):
            for j in range(i + 1, N):
                if j + 1 >= N:
                    continue
                dx_ji = x[j] - x[i]
                dy_ji = y[j] - y[i]
                dx_ki = x[j+1:] - x[i]
                dy_ki = y[j+1:] - y[i]
                cross = dx_ji * dy_ki - dx_ki * dy_ji
                tri_areas = 0.5 * np.abs(cross)
                if len(tri_areas) > 0:
                    areas.append(tri_areas)
                    
        if not areas:
            return 1e10
            
        all_areas = np.concatenate(areas)
        
        if len(all_areas) == 0 or np.any(all_areas < 1e-10):
            return 1e10
            
        # Soft min approximation
        beta = 50.0
        min_a = np.min(all_areas)
        
        # z_i = -beta * areas_i
        z = -beta * all_areas
        max_z = np.max(z)
        
        if max_z < -1000:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
            
        soft_min_val = -log_sum / beta
        score = soft_min_val / hull_area
        
        if score < 1e-6:
            return 1e10
            
        return -score

    def _refine_from_start(params0, time_budget):
        """Refine a starting point using NM then Powell, returning best found."""
        best_params = params0.copy()
        best_s = -_objective(best_params)
        
        # Nelder-Mead
        t_check = time.time()
        if t_check < deadline - max(5, time_budget * 0.4):
            try:
                res = minimize(
                    _objective, best_params, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                s = -res.fun
                if s > best_s:
                    best_s = s
                    best_params = res.x
            except Exception:
                pass
        
        # Powell
        t_check = time.time()
        if t_check < deadline - 5:
            try:
                res = minimize(
                    _objective, best_params, method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                s = -res.fun
                if s > best_s:
                    best_s = s
                    best_params = res.x
            except Exception:
                pass
        
        return best_params, best_s

    # Diverse initializations in parameter space (angle, radius)
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon (unit radius)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    inits.append(np.column_stack([angles, radii]).flatten())
    
    # 2. Jittered angles, unit radius
    angles_j = angles + rng.normal(0, 0.05, N)
    inits.append(np.column_stack([angles_j, radii]).flatten())
    
    # 3. Varying radii (elliptical)
    radii_e = np.ones(N)
    radii_e[::2] = 1.2
    radii_e[1::2] = 0.8
    inits.append(np.column_stack([angles, radii_e]).flatten())
    
    # 4. Random radii, regular angles
    radii_r = 1.0 + rng.normal(0, 0.2, N)
    inits.append(np.column_stack([angles, radii_r]).flatten())
    
    # 5. Jittered angles and radii
    angles_j2 = angles + rng.normal(0, 0.1, N)
    radii_r2 = 1.0 + rng.normal(0, 0.3, N)
    inits.append(np.column_stack([angles_j2, radii_r2]).flatten())
    
    best_result_params = inits[0].copy()
    best_score_val = -_objective(best_result_params)
    
    # Allocate time budget per init
    n_inits = len(inits)
    time_per_init = max(10.0, (deadline - start_time - 10) / n_inits)
    
    for idx, x0 in enumerate(inits):
        if time.time() >= deadline - 15:
            break
        
        t0 = time.time()
        params_refined, s_refined = _refine_from_start(x0, time_per_init)
        
        if s_refined > best_score_val:
            best_score_val = s_refined
            best_result_params = params_refined
            
        if time.time() >= deadline - 15:
            break
    
    # Final polish on the best result
    if time.time() < deadline - 8:
        try:
            res_final = minimize(
                _objective, best_result_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res_final.fun
            if s > best_score_val:
                best_score_val = s
                best_result_params = res_final.x
        except Exception:
            pass
            
    if time.time() < deadline - 5:
        try:
            res_final2 = minimize(
                _objective, best_result_params, method='Powell',
                options={'maxiter': 1500, 'xtol': 1e-11, 'ftol': 1e-13}
            )
            s = -res_final2.fun
            if s > best_score_val:
                best_score_val = s
                best_result_params = res_final2.x
        except Exception:
            pass
    
    # Final normalization
    final_points = _points_from_params(best_result_params)
    
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if something went wrong
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)]).copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)]).copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        
    return final_points
# EVOLVE-BLOCK-END
