"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed where possible, but N=14 is small enough for simple loops.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Iterate over all pairs (i, j) and compute cross product with all k > j
    # This avoids redundant calculations and keeps memory low.
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # Vectors from i to all k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            # Cross product magnitude gives 2 * area
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for minimization.
    We want to maximize score, so we minimize -score.
    """
    points = points_flat.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def _soft_min_objective(points_flat, beta=100.0):
    """
    Objective function using a soft-min approximation for the triangle areas.
    This makes the objective smoother, potentially helping gradient-based methods
    or allowing for more stable convergence.
    """
    points = points_flat.reshape(N, 2)
    
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
        
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, we can use log-sum-exp
    # log(sum(exp(-beta * areas))) = -beta * min(areas) + log(sum(exp(beta*(min(areas)-areas))))
    min_a = np.min(all_areas)
    # If min_a is very small, the soft min will be close to min_a
    # We want to maximize min_a / hull_area, so minimize -soft_min / hull_area
    
    # Using log-sum-exp trick for stability
    # Let z_i = -beta * areas_i
    # sum(exp(z_i)) = exp(max_z) * sum(exp(z_i - max_z))
    # log_sum = max_z + log(sum(exp(z_i - max_z)))
    
    z = -beta * all_areas
    max_z = np.max(z)
    if max_z < -1000: # Underflow risk, though exp(-1000) is tiny
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    soft_min_val = -log_sum / beta
    
    score = soft_min_val / hull_area
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Initialize with a regular 14-gon.
    2. Use Nelder-Mead for robust local refinement.
    3. Use Powell's method for a final polish.
    4. Add a second round of Nelder-Mead with tighter tolerances or different start if time permits.
    5. Normalize the result to unit area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = reg_points.flatten()
    
    best_result = x0.copy()
    best_score_val = _score(best_result.reshape(N, 2))
    
    # 2. Nelder-Mead Refinement (Round 1)
    if time.time() < deadline - 40:
        try:
            res_nm = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
            current_score = -res_nm.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result = res_nm.x
        except Exception:
            pass
    
    # 3. Powell's Method Polish
    if time.time() < deadline - 20:
        try:
            res_powell = minimize(
                _objective, 
                best_result, 
                method='Powell', 
                options={
                    'maxiter': 2000, 
                    'xtol': 1e-10, 
                    'ftol': 1e-12
                }
            )
            current_score = -res_powell.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result = res_powell.x
        except Exception:
            pass
            
    # 4. Nelder-Mead Refinement (Round 2) - Try to escape local minima or polish further
    if time.time() < deadline - 10:
        # Slightly perturb the best result to see if we can improve
        # Or just run NM again with tighter settings
        try:
            res_nm2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            current_score = -res_nm2.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result = res_nm2.x
        except Exception:
            pass

    # 5. Try soft-min objective if time remains? 
    # The soft-min objective is smoother, might allow gradient-based methods or better NM convergence.
    # However, switching objectives mid-run is risky. Let's stick to the hard min for consistency
    # but maybe use soft-min for a separate restart if we had more time. 
    # Given the time constraints, the above multi-stage NM/Powell is a solid refinement strategy.
    
    # 6. Final Verification and Normalization
    final_points = best_result.reshape(N, 2)
    
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        final_points = reg_points.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        # Degenerate case, return regular polygon normalized
        final_points = reg_points.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
