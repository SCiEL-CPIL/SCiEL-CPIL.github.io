"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the packing radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    if n == 1:
        return float(1.0 - np.linalg.norm(points[0]))
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Boundary distance
    norms = np.sqrt(np.sum(points * points, axis=1))
    min_bound = np.min(1.0 - norms)
    
    return float(min(min_pair, min_bound))

def _build_points_from_params(params, ring_config):
    """
    Build points from parametric representation.
    
    params: array of [r1, r2, ..., rk, theta1, theta2, ..., theta_k]
    where ri are ring radii and theta_i are angular offsets for each ring.
    
    ring_config: list of integers specifying how many points on each ring.
    The first ring (if present) can include the center point.
    """
    k = len(ring_config)
    radii = params[:k]
    angles = params[k:]
    
    points = []
    for i, count in enumerate(ring_config):
        if count == 0:
            continue
        r = radii[i]
        theta_offset = angles[i]
        for j in range(count):
            angle = theta_offset + 2.0 * np.pi * j / count
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
    
    return np.array(points[:N])

def _parametric_objective(params, ring_config):
    """
    Objective function for scipy minimize (minimizes negative radius).
    """
    points = _build_points_from_params(params, ring_config)
    if len(points) != N:
        return 1e6
    
    # Check if any point is outside the unit circle
    norms = np.sqrt(np.sum(points * points, axis=1))
    if np.any(norms > 1.0):
        return 1e6
    
    r = _compute_score(points)
    return -r

def _soft_min_surrogate(points, alpha=50.0):
    """
    Smooth approximation of min using log-sum-exp.
    Returns a smooth surrogate for the minimum of pairwise distances and boundary distances.
    """
    n = len(points)
    if n < 2:
        return 1.0 - np.linalg.norm(points[0])
    
    # Pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Boundary squared distances
    norms_sq = np.sum(points * points, axis=1)
    bound_sq = (1.0 - np.sqrt(norms_sq)) ** 2
    
    # Use soft min on squared distances, then take sqrt
    all_sq = np.concatenate([dist_sq.flatten(), bound_sq])
    all_sq = all_sq[all_sq > 1e-12]  # Remove zeros/infinities
    
    if len(all_sq) == 0:
        return 0.0
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * x)))
    max_val = np.max(all_sq)
    log_sum = np.log(np.sum(np.exp(-alpha * (all_sq - max_val))))
    soft_min_sq = max_val - np.log(len(all_sq)) - log_sum / alpha
    
    return float(np.sqrt(max(0.0, soft_min_sq)))

def _smooth_objective(params, ring_config, alpha=50.0):
    """
    Smooth objective using soft-min surrogate.
    """
    points = _build_points_from_params(params, ring_config)
    if len(points) != N:
        return 1e6
    
    norms = np.sqrt(np.sum(points * points, axis=1))
    if np.any(norms > 1.0):
        return 1e6
    
    return -_soft_min_surrogate(points, alpha)

def _polish_with_smooth(params, ring_config, time_budget=10.0):
    """
    Polish using smooth objective with multiple alpha values.
    """
    current_params = params.copy()
    start_time = time.time()
    
    # Start with larger alpha (smoother) and decrease
    alphas = [20.0, 30.0, 50.0, 100.0, 200.0]
    
    for alpha in alphas:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _smooth_objective(p, ring_config, alpha)
        
        try:
            result = minimize(obj, current_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            current_params = result.x
        except Exception:
            pass
    
    return current_params

def _exact_polish(points, time_budget=15.0):
    """
    Final exact polish using Nelder-Mead on the true objective.
    """
    # Convert to parametric form by estimating ring structure
    # For simplicity, do direct coordinate optimization
    def obj(p):
        pts = p.reshape(-1, 2)
        norms = np.sqrt(np.sum(pts * pts, axis=1))
        if np.any(norms > 1.0):
            return 1e6
        return -_compute_score(pts)
    
    x0 = points.flatten()
    start_time = time.time()
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True})
        best_pts = result.x.reshape(-1, 2)
        
        # Validate
        norms = np.sqrt(np.sum(best_pts * best_pts, axis=1))
        if np.any(norms > 1.0) or len(best_pts) != N:
            return points, _compute_score(points)
        
        r = _compute_score(best_pts)
        return best_pts, r
    except Exception:
        return points, _compute_score(points)

def _try_ring_config(ring_config, initial_params=None, time_budget=10.0):
    """
    Optimize a given ring configuration.
    """
    k = len(ring_config)
    
    if initial_params is None:
        # Initialize with reasonable radii
        radii = np.linspace(0.15, 0.85, k)
        angles = np.zeros(k)
        params = np.concatenate([radii, angles])
    else:
        params = initial_params.copy()
    
    # First, optimize with smooth objective
    polished_params = _polish_with_smooth(params, ring_config, time_budget=time_budget * 0.6)
    
    # Then, exact polish
    points = _build_points_from_params(polished_params, ring_config)
    if len(points) != N:
        return None, -1.0
    
    final_points, final_r = _exact_polish(points, time_budget=time_budget * 0.4)
    
    return final_points, final_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_r = -1.0
    
    # Try various ring configurations
    ring_configs = [
        [1, 6, 12, 18, 13],   # 1+6+12+18+13=50
        [1, 6, 12, 19, 12],   # 1+6+12+19+12=50
        [1, 6, 13, 18, 12],   # 1+6+13+18+12=50
        [1, 7, 12, 18, 12],   # 1+7+12+18+12=50
        [1, 6, 12, 17, 14],   # 1+6+12+17+14=50
        [1, 6, 13, 17, 13],   # 1+6+13+17+13=50
        [1, 8, 12, 17, 12],   # 1+8+12+17+12=50
        [1, 6, 14, 18, 11],   # 1+6+14+18+11=50
        [1, 7, 13, 17, 12],   # 1+7+13+17+12=50
        [1, 6, 12, 18, 12, 1], # 1+6+12+18+12+1=50
    ]
    
    num_configs = len(ring_configs)
    
    for i, config in enumerate(ring_configs):
        if time.time() > deadline - 20.0:
            break
        
        # Check if config sums to N
        if sum(config) != N:
            continue
        
        remaining = deadline - time.time() - 15.0
        if remaining < 5.0:
            break
        
        tb = min(15.0, remaining / max(1, num_configs - i))
        
        try:
            pts, r = _try_ring_config(config, time_budget=tb)
            if pts is not None and r > best_r:
                best_r = r
                best_points = pts
        except Exception:
            pass
    
    # If no good result, fall back to a simple ring initialization
    if best_points is None:
        # Use the first valid config
        for config in ring_configs:
            if sum(config) == N:
                k = len(config)
                radii = np.linspace(0.15, 0.85, k)
                angles = np.zeros(k)
                params = np.concatenate([radii, angles])
                best_points = _build_points_from_params(params, config)
                best_r = _compute_score(best_points)
                break
    
    # Final polish if time permits
    if best_points is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        final_pts, final_r = _exact_polish(best_points, time_budget=max(3.0, remaining))
        if final_r > best_r:
            best_r = final_r
            best_points = final_pts
    
    return best_points

# EVOLVE-BLOCK-END
