"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
# i < j < k
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _smooth_min_area(pts, beta):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    if min_a < 1e-10:
         return 0.0
         
    y = -beta * (areas - min_a)
    # Clip to prevent underflow issues in exp, though y <= 0 so exp(y) <= 1
    # If y is very negative, exp(y) is 0. This is fine for sum.
    # However, if min_a is very small and areas vary, y can be large negative.
    # np.exp handles this by returning 0.
    sum_exp = np.sum(np.exp(y))
    if sum_exp == 0:
        return min_a
    log_sum_exp = np.log(sum_exp)
    
    soft_min = min_a - (1.0 / beta) * log_sum_exp
    return soft_min

def _objective_smooth(params, beta):
    """
    Objective: - (smooth_min_area / hull_area)
    """
    pts = params.reshape(N, 2)
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
    
    s_min = _smooth_min_area(pts, beta)
    if s_min <= 0:
        return 1e9
        
    return -(s_min / h_area)

def _generate_starts(rng):
    """Generate diverse initial points for multi-start optimization."""
    starts = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(p_reg)
    
    # 2. Jittered regular polygons
    for _ in range(5):
        noise = rng.normal(0, 0.2, size=(N, 2))
        p_jit = p_reg + noise
        starts.append(p_jit)
        
    # 3. Random convex-like starts (sorted angles)
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.5, N)
        p_rand = np.column_stack((radii * np.cos(angles), rng.uniform(0.5, 1.5, N) * np.sin(angles)))
        # Ensure convexity roughly by sorting by angle, but radii vary
        # This is a heuristic start
        p_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(p_rand)
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # --- Phase 1: Global Search with Differential Evolution ---
    # We optimize in Cartesian coordinates directly to allow non-convex hulls if they help (though unlikely for max min area).
    # Bounds: keep points within [-2, 2] to avoid degenerate huge hulls or tiny clusters.
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Use a moderate beta for the smooth objective during global search
    beta_global = 20.0
    
    if (time.time() - start_time) < deadline - 60.0:
        try:
            de_res = differential_evolution(
                _objective_smooth,
                bounds,
                args=(beta_global,),
                maxiter=80,
                popsize=20,
                tol=1e-4,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if de_res.fun < 1e8:
                de_pts = de_res.x.reshape(N, 2)
                de_score = _compute_score(de_pts)
                
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts.copy()
                    
        except Exception:
            pass
    
    # --- Phase 2: Local Refinement with Multi-Start BFGS ---
    # If we found a good point from DE, use it as a primary start.
    # Also try diverse starts to ensure we don't get stuck in a local optimum of the smooth approximation.
    
    starts_to_try = []
    if best_pts is not None:
        starts_to_try.append(best_pts)
    
    # Add diverse starts
    diverse_starts = _generate_starts(rng)
    for p in diverse_starts:
        starts_to_try.append(p)
    
    # Schedule of increasing beta for sharper minimum approximation
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_best_for_phase2 = best_pts.copy() if best_pts is not None else None
    
    for beta in betas:
        if (time.time() - start_time) > deadline - 20.0:
            break
            
        # If we have a current best, prioritize it
        if current_best_for_phase2 is not None:
            # Try the current best first with this beta
            try:
                res = minimize(
                    _objective_smooth, 
                    current_best_for_phase2.flatten(), 
                    args=(beta,),
                    method='BFGS', 
                    options={'maxiter': 100, 'gtol': 1e-7}
                )
                if res.fun < 1e8:
                    new_pts = res.x.reshape(N, 2)
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        current_best_for_phase2 = best_pts.copy()
            except Exception:
                pass
            
            # Then try a few diverse starts if time permits
            if (time.time() - start_time) > deadline - 15.0:
                continue
                
            for p0 in diverse_starts[:3]: # Limit to 3 diverse starts per beta level
                if (time.time() - start_time) > deadline - 10.0:
                    break
                try:
                    res = minimize(
                        _objective_smooth, 
                        p0.flatten(), 
                        args=(beta,),
                        method='BFGS', 
                        options={'maxiter': 50, 'gtol': 1e-6}
                    )
                    if res.fun < 1e8:
                        new_pts = res.x.reshape(N, 2)
                        new_score = _compute_score(new_pts)
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = new_pts.copy()
                            current_best_for_phase2 = best_pts.copy()
                except Exception:
                    continue
        else:
            # If no best yet, try all diverse starts
            for p0 in diverse_starts:
                if (time.time() - start_time) > deadline - 10.0:
                    break
                try:
                    res = minimize(
                        _objective_smooth, 
                        p0.flatten(), 
                        args=(beta,),
                        method='BFGS', 
                        options={'maxiter': 100, 'gtol': 1e-6}
                    )
                    if res.fun < 1e8:
                        new_pts = res.x.reshape(N, 2)
                        new_score = _compute_score(new_pts)
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = new_pts.copy()
                            current_best_for_phase2 = best_pts.copy()
                except Exception:
                    continue

    # --- Phase 3: Final Polish with Nelder-Mead on Exact Objective ---
    if best_pts is not None and (time.time() - start_time) < deadline - 5.0:
        try:
            def _exact_obj(params):
                pts = params.reshape(N, 2)
                s = _compute_score(pts)
                return -s if s > 0 else 1e9
            
            res_polish = minimize(
                _exact_obj, 
                best_pts.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_polish.fun < 1e8:
                polish_pts = res_polish.x.reshape(N, 2)
                polish_score = _compute_score(polish_pts)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_pts = polish_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts

# EVOLVE-BLOCK-END
