"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

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
    # i < j < k
    idx_i, idx_j, idx_k = np.indices((N, N, N)).reshape(3, -1)
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i = idx_i[mask]
    j = idx_j[mask]
    k = idx_k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _logsumexp_min_obj(params, tau):
    """
    Smooth surrogate for the negative minimum triangle area ratio.
    params: (2N,) array of [x_1, y_1, ..., x_N, y_N]
    tau: sharpness parameter for logsumexp approximation of min
    """
    pts = params.reshape(N, 2)
    
    # Compute Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e9
    except Exception:
        return 1e9

    # Compute all triangle areas
    idx_i, idx_j, idx_k = np.indices((N, N, N)).reshape(3, -1)
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i = idx_i[mask]
    j = idx_j[mask]
    k = idx_k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    
    # We want to maximize min(ratios).
    # Minimize -min(ratios).
    # Approx -min(y) with (1/tau) * log(sum(exp(-tau * y)))
    # Note: for large tau, this converges to -min(y).
    
    # Numerical stability: subtract max of exponents
    exponents = -tau * ratios
    max_exp = np.max(exponents)
    if max_exp > 100:
        # Overflow protection, though ratios are small
        max_exp = 100
    
    sum_exp = np.sum(np.exp(exponents - max_exp))
    log_sum = max_exp + np.log(sum_exp)
    
    return log_sum / tau

def _smooth_optimize(p0, time_limit):
    """
    Optimize using BFGS on the smooth logsumexp objective.
    """
    start_time = time.time()
    
    # Annealing schedule for tau
    taus = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_pts = p0.copy()
    best_score = _compute_score(current_pts)
    
    for tau in taus:
        if (time.time() - start_time) > time_limit:
            break
            
        def objective(params):
            return _logsumexp_min_obj(params, tau)
        
        try:
            res = minimize(
                objective, 
                current_pts.flatten(), 
                method='BFGS', 
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            
            if res.success:
                new_pts = res.x.reshape(N, 2)
                new_score = _compute_score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    current_pts = new_pts.copy()
                    
        except Exception:
            continue
            
    return current_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Generate diverse starts
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    
    starts = [p_reg]
    
    # 2. Jittered regular polygons
    for _ in range(10):
        noise = rng.normal(0, 0.1, size=(N, 2))
        p_jit = p_reg + noise
        starts.append(p_jit)
        
    # 3. Random convex-like starts
    # Generate random angles, sort them, random radii
    for _ in range(10):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.8, 1.2, N)
        p_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(p_rand)
        
    # Optimize each start
    for p0 in starts:
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        try:
            pts_opt, score_opt = _smooth_optimize(p0, time_limit - (time.time() - start_time))
            
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
                
        except Exception:
            continue
            
    if best_pts is None:
        return p_reg
        
    return best_pts

# EVOLVE-BLOCK-END
