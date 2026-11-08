"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """
    Return array of all triangle areas (N choose 3) for N points.
    Vectorized inner loop for speed.
    """
    n = len(points)
    if n < 3:
        return np.array([0.0])
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Points k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product of (p_j - p_i) and (p_k - p_i)
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _min_triangle_area(points):
    """Compute the minimum triangle area."""
    areas = _all_triangle_areas(points)
    return np.min(areas) if len(areas) > 0 else 0.0

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    if min_area <= 1e-12:
        return 0.0
    return min_area / hull_a

def _soft_obj(pts_flat, beta=100.0):
    """
    Soft objective: hull_area / soft_min(triangle areas).
    Minimizing this maximizes soft_min / hull_area.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    # Check for degenerate points (very close together)
    # If any point is too close to another, the min area will be tiny.
    # The soft min will handle this, but we add a check for robustness.
    
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    # Soft min calculation
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to minimize hull_area / soft_min(areas)
    # Which is equivalent to maximizing soft_min(areas) / hull_area
    
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        # Avoid underflow/overflow issues, though exp(-inf) is 0
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / soft_min_val

def _hard_obj(pts_flat):
    """
    Hard objective: hull_area / min_triangle_area.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 1e-12:
        return pts / np.sqrt(hull_a)
    return pts

def _init_regular(n):
    """Regular polygon on unit circle."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)

def _init_perturbed(n, seed=42):
    """Regular polygon with random angular and radial perturbations."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n
    # Perturb angles
    angle_noise = rng.normal(0, 0.1, size=n)
    angles = angles + angle_noise
    # Perturb radii
    radii = 1.0 + rng.normal(0, 0.1, size=n)
    radii = np.clip(radii, 0.5, 1.5)
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)

def _init_mixed(n):
    """Mix of outer and inner points."""
    # For N=14, try 10 outer, 4 inner
    outer_n = 10
    inner_n = 4
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)

def construct():
    start_time = time.time()
    
    # Generate initial points
    inits = [
        _init_regular(N),
        _init_perturbed(N, seed=1),
        _init_perturbed(N, seed=2),
        _init_mixed(N),
    ]
    
    best_pts = None
    best_score = 0.0
    
    # Optimization settings
    # Nelder-Mead is a derivative-free method, good for this non-smooth-ish landscape
    # We use it on the smooth soft objective first, then the hard one.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        x0 = init.flatten()
        
        # Stage 1: Soft objective optimization
        # Use a moderate beta to balance smoothness and fidelity
        try:
            res_soft = minimize(
                _soft_obj, 
                x0, 
                method='Nelder-Mead',
                args=(100.0,),
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_soft.x is not None:
                cand = res_soft.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
            
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Stage 2: Hard objective optimization (Polish)
        # Start from the result of Stage 1
        x1 = res_soft.x if 'res_soft' in locals() and res_soft.x is not None else x0
        try:
            res_hard = minimize(
                _hard_obj, 
                x1, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if res_hard.x is not None:
                cand = res_hard.x.reshape(N, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass

    # If we still have time, try a few more random perturbations of the best found
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        rng = np.random.RandomState(123)
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 2:
                break
            noise = rng.normal(0, 0.02, size=best_pts.shape)
            perturbed = best_pts + noise
            # Re-normalize
            perturbed = _normalize_unit_area(perturbed)
            
            # Quick polish
            x0 = perturbed.flatten()
            try:
                res = minimize(
                    _hard_obj, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if res.x is not None:
                    cand = res.x.reshape(N, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass

    # Fallback
    if best_pts is None:
        best_pts = _init_regular(N)
        
    return best_pts
# EVOLVE-BLOCK-END
