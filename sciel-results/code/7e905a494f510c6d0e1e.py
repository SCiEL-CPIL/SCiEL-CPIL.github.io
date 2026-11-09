import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute triangle indices for the full 11-point set
# i < j < k
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)
idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def params_to_points(params):
    """
    Reconstruct the full 11x2 points array from the reduced 11x1 parameters.
    params: [w, u0, v0, u1, v1, ..., u4, v4]
    w: y-coordinate of the central point (scaled by H)
    u_i, v_i: parameters for the i-th pair in the right half
    """
    w = params[0]
    cy = w * H
    pts = np.empty((11, 2))
    pts[0] = [0.5, cy]
    
    for i in range(5):
        u = params[1 + 2*i]
        v = params[2 + 2*i]
        # Map [0,1] to the right half of the triangle
        x = 0.5 + 0.5 * u
        y = v * SQRT3 * (1.0 - x)
        
        pts[1 + i] = [x, y]
        pts[6 + i] = [1.0 - x, y]
        
    return pts

def calculate_areas(points):
    """
    Calculate areas of all triangles formed by the points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def soft_min_objective(params, k=30.0):
    """
    Smooth surrogate for the minimum area.
    Minimizing this function approximates maximizing the minimum area.
    """
    points = params_to_points(params)
    areas = calculate_areas(points)
    
    # Soft min: (1/k) * log(sum(exp(-k * areas)))
    # This is stable for small positive areas.
    sum_exp = np.sum(np.exp(-k * areas))
    return (1.0 / k) * np.log(sum_exp)

def exact_min_objective(params):
    """
    Exact objective: -min(area).
    Includes penalty if points are outside the triangle (should not happen with our parameterization, 
    but floating point errors or NM steps might cause issues if we didn't use parameterization. 
    With parameterization, points are always inside. 
    However, NM does not respect bounds on parameters, so it might go to negative u/v or >1.
    If u < 0 or u > 1 or v < 0 or v > 1, points might be outside.
    So we MUST check validity in exact_min_objective if used with NM.
    """
    points = params_to_points(params)
    
    # Check if parameters are within [0,1] range roughly
    # If NM steps outside, the geometry is invalid.
    # We can just return a large penalty if any parameter is out of [0,1]
    if np.any(params < -1e-9) or np.any(params > 1 + 1e-9):
        return 100.0
        
    # Check geometric validity (redundant if params in [0,1], but safe)
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -1e-9) or np.any(y > SQRT3 * x + 1e-9) or np.any(y > SQRT3 * (1 - x) + 1e-9):
        return 100.0

    areas = calculate_areas(points)
    return -np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 11 variables: 1 for center y, 10 for 5 pairs (u,v)
    bounds = [(0.0, 1.0)] * 11
    
    best_params = None
    best_score = -float('inf')
    
    # Phase 1: Differential Evolution
    try:
        res_de = differential_evolution(
            soft_min_objective,
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-5,
            seed=42,
            polish=False,
            init='sobol',
            args=(30.0,)
        )
        best_params = res_de.x
        # The score is -min_area, so we want to maximize that.
        # res_de.fun is soft_min, which is approx -min_area.
        # We just keep the params.
        
    except Exception:
        # Fallback to random
        best_params = np.random.rand(11)
        
    # Phase 2: L-BFGS-B Refinement (Smooth)
    if time.time() < deadline - 30:
        try:
            res_lbfgs = minimize(
                soft_min_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10},
                args=(30.0,)
            )
            # Check if it improved the soft-min score
            if res_lbfgs.fun < soft_min_objective(best_params, 30.0):
                best_params = res_lbfgs.x
        except Exception:
            pass

    # Phase 3: Nelder-Mead Refinement (Exact)
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                exact_min_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            # Check if result is valid (not penalized)
            if res_nm.fun < 50.0:
                best_params = res_nm.x
        except Exception:
            pass
            
    # Final Polish with high precision if time permits
    if time.time() < deadline - 1:
        try:
            res_nm2 = minimize(
                exact_min_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm2.fun < 50.0 and res_nm2.fun < exact_min_objective(best_params):
                best_params = res_nm2.x
        except Exception:
            pass

    final_points = params_to_points(best_params)
    
    return final_points
