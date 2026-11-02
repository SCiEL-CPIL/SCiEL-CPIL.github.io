import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    if n < 3:
        return float('inf')
    
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n - 2):
        dx_j = x[i+1:] - x[i]
        dy_j = y[i+1:] - y[i]
        
        for j_idx in range(len(dx_j)):
            j = i + 1 + j_idx
            dx_ij = dx_j[j_idx]
            dy_ij = dy_j[j_idx]
            
            if j + 1 >= n:
                continue
            
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            
            cross = dx_ij * dy_ik - dx_ik * dy_ij
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    if min_area < 1e-10:
                        return min_area
                        
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

def _build_symmetric_points(params):
    """
    Reconstruct the 14 points from 7 parameters (14 coordinates).
    params: (7, 2) array of the first 7 points.
    The next 7 points are -params (central symmetry).
    """
    p1 = params
    p2 = -params
    points = np.vstack([p1, p2])
    return points

def _objective_soft(params, beta=100.0):
    """
    Objective function to minimize: -soft_min(triangle_areas / hull_area).
    Uses a smooth approximation to the minimum to allow gradient-based methods.
    """
    points = _build_symmetric_points(params)
    
    # Check for degeneracy
    if np.any(np.linalg.norm(points, axis=1) < 1e-6):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
    # Compute all triangle areas
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    areas = []
    for i in range(n - 2):
        dx_j = x[i+1:] - x[i]
        dy_j = y[i+1:] - y[i]
        
        for j_idx in range(len(dx_j)):
            j = i + 1 + j_idx
            dx_ij = dx_j[j_idx]
            dy_ij = dy_j[j_idx]
            
            if j + 1 >= n:
                continue
            
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            
            cross = dx_ij * dy_ik - dx_ik * dy_ij
            tri_areas = 0.5 * np.abs(cross)
            areas.extend(tri_areas.tolist())
            
    if not areas:
        return 1e10
        
    areas_arr = np.array(areas)
    
    # Normalize by hull area
    normalized_areas = areas_arr / hull_area
    
    # Soft min: -log-sum-exp(-beta * x) / beta
    # To avoid overflow, we shift by the max
    max_val = np.max(normalized_areas)
    if max_val < 1e-10:
        return 1e10
        
    log_sum_exp = np.log(np.sum(np.exp(-beta * (normalized_areas - max_val))))
    soft_min = max_val - (1.0 / beta) * log_sum_exp
    
    # We want to maximize the min area, so we minimize -soft_min
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score_val = 0.0
    best_params = None
    
    # Initial guess: Regular 14-gon split into 7 symmetric pairs
    # Angles for 7 points in the first half
    angles = np.linspace(0, np.pi, N_HALF, endpoint=False)
    radii = np.ones(N_HALF)
    init_params = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Evaluate initial
    init_points = _build_symmetric_points(init_params)
    init_score = _score(init_points)
    best_score_val = init_score
    best_params = init_params.copy()
    
    # 1. Global search with Differential Evolution in the 14-dim parameter space
    # Bounds: x in [-1, 1], y in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N_HALF)
    
    if time.time() < deadline - 60:
        try:
            def de_objective(params):
                if time.time() > deadline - 5:
                    return 1e10
                return _objective_soft(params, beta=50.0)
            
            de_res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                atol=1e-8,
                init='sobol',
                update_interval=10,
                polish=False  # We will polish manually
            )
            
            if de_res.fun < 1e9:
                de_params = de_res.x
                de_points = _build_symmetric_points(de_params)
                de_score = _score(de_points)
                if de_score > best_score_val:
                    best_score_val = de_score
                    best_params = de_params.copy()
        except Exception:
            pass
            
    # 2. Local refinement with Nelder-Mead and Powell on the best candidate
    # Multi-start local search from best_params and a few perturbations
    
    local_starts = [best_params.copy()]
    
    # Add a few perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = best_params + rng.normal(0, 0.05, size=best_params.shape)
        local_starts.append(perturbed)
        
    # Also include the regular polygon start
    local_starts.append(init_params.copy())
    
    for start_p in local_starts:
        if time.time() > deadline - 10:
            break
            
        # Nelder-Mead
        try:
            nm_res = minimize(
                _objective_soft,
                start_p,
                args=(50.0,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if nm_res.fun < 1e9:
                nm_params = nm_res.x
                nm_points = _build_symmetric_points(nm_params)
                nm_score = _score(nm_points)
                if nm_score > best_score_val:
                    best_score_val = nm_score
                    best_params = nm_params.copy()
        except Exception:
            pass
            
        # Powell
        if time.time() > deadline - 10:
            continue
        try:
            powell_res = minimize(
                _objective_soft,
                best_params,
                args=(50.0,),
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if powell_res.fun < 1e9:
                powell_params = powell_res.x
                powell_points = _build_symmetric_points(powell_params)
                powell_score = _score(powell_points)
                if powell_score > best_score_val:
                    best_score_val = powell_score
                    best_params = powell_params.copy()
        except Exception:
            pass
            
    # 3. Final high-precision polish with L-BFGS-B using a softer beta for smoothness
    if time.time() < deadline - 5:
        try:
            lbfgs_res = minimize(
                _objective_soft,
                best_params,
                args=(20.0,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if lbfgs_res.fun < 1e9:
                lbfgs_params = lbfgs_res.x
                lbfgs_points = _build_symmetric_points(lbfgs_params)
                lbfgs_score = _score(lbfgs_points)
                if lbfgs_score > best_score_val:
                    best_score_val = lbfgs_score
                    best_params = lbfgs_params.copy()
        except Exception:
            pass
            
    # Reconstruct final points
    if best_params is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        final_points = _build_symmetric_points(best_params)
        
    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        
    return final_points
