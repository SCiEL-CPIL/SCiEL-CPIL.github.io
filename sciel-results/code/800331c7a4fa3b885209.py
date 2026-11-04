"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
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

def _soft_min_score(pts, beta):
    """
    Compute a smooth approximation of the minimum triangle area / hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
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
    sum_exp = np.sum(np.exp(y))
    if sum_exp == 0:
        return min_a / hull_area
    log_sum_exp = np.log(sum_exp)
    
    soft_min = min_a - (1.0 / beta) * log_sum_exp
    if soft_min <= 0:
        return 0.0
        
    return soft_min / hull_area

def _polar_to_pts(params):
    """
    Convert polar parameters to Cartesian points.
    params: [theta_0, ..., theta_{N-1}, r_0, ..., r_{N-1}]
    - theta_i are angular steps (log-space), ensuring sorted angles
    - r_i are radial distances (log-space), with monotonic increase enforced
    """
    # Extract parameters
    theta_steps = np.exp(params[:N])  # Positive angular steps
    r_base = np.exp(params[N:])  # Base radial distances
    
    # Ensure angles are sorted and cover 2*pi
    angles = np.cumsum(theta_steps)
    total_angle = angles[-1]
    angles = angles / total_angle * 2 * np.pi
    
    # Enforce monotonic increase in radial distance
    # Use softplus-like function: r_i = r_{i-1} + softplus(delta_i)
    # But we want to allow the optimizer to control this
    # Let's use: r_i = base_r + cumsum(softplus(deltas))
    # Actually, let's just use the raw values and let the optimizer figure it out
    # The key is that the representation is smooth
    
    radii = r_base
    # Optionally enforce monotonicity with a soft constraint
    # For now, just use the values directly
    
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return pts

def _polar_objective(params, beta):
    """
    Objective for polar parameterization with monotonic radial distance.
    """
    pts = _polar_to_pts(params)
    
    # Check if points are valid (non-degenerate)
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e9
    
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
        
    s_min = _soft_min_score(pts, beta)
    if s_min <= 0:
        return 1e9
        
    return -s_min

def _generate_polar_starts(rng, n_starts):
    """
    Generate diverse initial polar parameterizations.
    """
    starts = []
    
    for i in range(n_starts):
        # Base: regular polygon
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base_radii = np.ones(N)
        
        # Add perturbations
        if i == 0:
            # Regular polygon
            theta_steps = np.full(N, 2 * np.pi / N)
            r_base = np.ones(N)
        elif i == 1:
            # Slightly perturbed angles
            theta_steps = (2 * np.pi / N) * (1 + 0.1 * rng.standard_normal(N))
            theta_steps = np.abs(theta_steps) + 0.01
            r_base = 1.0 + 0.05 * rng.standard_normal(N)
        elif i == 2:
            # Perturbed radii
            theta_steps = np.full(N, 2 * np.pi / N)
            r_base = 1.0 + 0.2 * rng.standard_normal(N)
            r_base = np.abs(r_base) + 0.1
        else:
            # Random
            theta_steps = np.abs(rng.standard_normal(N)) + 0.1
            r_base = np.abs(rng.standard_normal(N)) + 0.5
        
        params = np.concatenate([np.log(theta_steps), np.log(r_base)])
        starts.append(params)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Multi-start L-BFGS-B on polar parameterization
    n_starts = 8
    starts = _generate_polar_starts(rng, n_starts)
    
    for start_idx, start_params in enumerate(starts):
        if (time.time() - start_time) > deadline - 15.0:
            break
        
        try:
            # Use L-BFGS-B with smooth objective
            res = minimize(
                _polar_objective,
                start_params,
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if res.fun < 1e8:
                pts = _polar_to_pts(res.x)
                score = _compute_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
                    # If we have a good score, do a refinement
                    if best_score > 0.01 and (time.time() - start_time) < deadline - 10.0:
                        # Refine with higher beta
                        res2 = minimize(
                            _polar_objective,
                            res.x,
                            args=(100.0,),
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                        )
                        if res2.fun < 1e8:
                            pts2 = _polar_to_pts(res2.x)
                            score2 = _compute_score(pts2)
                            if score2 > best_score:
                                best_score = score2
                                best_pts = pts2.copy()
        except Exception:
            continue
    
    # Phase 2: Cartesian refinement with Nelder-Mead
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
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
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
