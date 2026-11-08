import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
N_OUTER = 7
N_INNER = 6

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _smooth_objective(params, beta=50.0):
    """
    Smooth objective function for coordinate-based optimization.
    Uses a soft minimum approximation to make the objective differentiable.
    params: 2*N coordinates (x1,y1,x2,y2,...,xN,yN)
    """
    try:
        pts = params.reshape(N, 2)
        
        # Normalize: center at centroid, scale so hull area = 1
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # Compute hull area of centered points
        try:
            hull = ConvexHull(pts_centered)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e10
        except Exception:
            return 1e10
        
        # Normalize to unit hull area
        scale = np.sqrt(1.0 / hull_area)
        pts_norm = pts_centered * scale
        
        # Compute all triangle areas
        areas = _all_triangle_areas(pts_norm)
        
        # Soft minimum: -log(sum(exp(-beta * area))) / beta approximates min
        # Use log-sum-exp trick for numerical stability
        min_val = np.min(areas)
        shifted = areas - min_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) 
        soft_min = min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta
        
        # We want to MAXIMIZE soft_min, so return negative
        return -soft_min
        
    except Exception:
        return 1e10

def _generate_initial_points(seed=42):
    """Generate diverse initial point configurations."""
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Regular polygon with slight noise
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
    radii = np.ones(N) + rng.uniform(-0.1, 0.1, N)
    pts1 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Strategy 2: Two concentric rings
    angles_out = np.linspace(0, 2 * np.pi, N_OUTER, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, N_INNER, endpoint=False) + np.pi / N_INNER
    radii_out = np.ones(N_OUTER)
    radii_in = 0.5 * np.ones(N_INNER)
    pts_out = np.column_stack((radii_out * np.cos(angles_out), radii_out * np.sin(angles_out)))
    pts_in = np.column_stack((radii_in * np.cos(angles_in), radii_in * np.sin(angles_in)))
    pts2 = np.vstack([pts_out, pts_in])
    
    # Strategy 3: Random points in unit circle
    angles3 = rng.uniform(0, 2 * np.pi, N)
    radii3 = rng.uniform(0.3, 1.0, N)
    pts3 = np.column_stack((radii3 * np.cos(angles3), radii3 * np.sin(angles3)))
    
    # Strategy 4: Fibonacci sphere-like distribution projected to 2D
    golden_angle = np.pi * (3 - np.sqrt(5))
    angles4 = np.arange(N) * golden_angle
    radii4 = np.sqrt(np.linspace(0.2, 1.0, N))
    pts4 = np.column_stack((radii4 * np.cos(angles4), radii4 * np.sin(angles4)))
    
    return [pts1, pts2, pts3, pts4]

def construct():
    t0 = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial configurations
    initial_configs = _generate_initial_points()
    
    # For each initial config, run local optimization
    for init_idx, pts_init in enumerate(initial_configs):
        if time.time() - t0 > deadline - 10:
            break
        
        # Flatten to parameter vector
        params_init = pts_init.flatten()
        
        # Try multiple beta values for the soft minimum
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - t0 > deadline - 5:
                break
            
            def obj(params, b=beta):
                return _smooth_objective(params, b)
            
            try:
                # Use L-BFGS-B with gradient approximation
                res = minimize(
                    obj,
                    params_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.fun < 1e9:  # Valid result
                    pts_result = res.x.reshape(N, 2)
                    score = _exact_score(pts_result)
                    
                    if score > best_score:
                        best_score = score
                        best_points = pts_result.copy()
                        
            except Exception:
                pass
        
        # Also try Nelder-Mead as fallback
        if time.time() - t0 > deadline - 5:
            break
            
        try:
            res_nm = minimize(
                lambda p: _smooth_objective(p, 50.0),
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < 1e9:
                pts_result = res_nm.x.reshape(N, 2)
                score = _exact_score(pts_result)
                
                if score > best_score:
                    best_score = score
                    best_points = pts_result.copy()
                    
        except Exception:
            pass
    
    # If we found a good solution, do a final polish with higher beta
    if best_points is not None and time.time() - t0 < deadline - 10:
        params_best = best_points.flatten()
        
        for beta in [200.0, 500.0]:
            if time.time() - t0 > deadline - 5:
                break
            
            try:
                res_polish = minimize(
                    lambda p: _smooth_objective(p, beta),
                    params_best,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if res_polish.fun < 1e9:
                    pts_polish = res_polish.x.reshape(N, 2)
                    score = _exact_score(pts_polish)
                    
                    if score > best_score:
                        best_score = score
                        best_points = pts_polish.copy()
                        
            except Exception:
                pass
    
    # Fallback if everything failed
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
