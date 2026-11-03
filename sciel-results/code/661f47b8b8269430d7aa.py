import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Compute pairwise distances efficiently
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def get_pair_indices():
    """Precompute upper triangle indices for pairwise distances."""
    return np.triu_indices(N, k=1)

def _project_inside(x):
    """Project points inside unit circle."""
    rad = np.linalg.norm(x, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        x = x.copy()
        x[mask] /= rad[mask, np.newaxis]
    return x

def _compute_all_dists(points):
    """Compute all pairwise distances and boundary distances."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    return dists, bound_dists

def _ls_residuals_per_pair(x_flat, pair_idx, k=50.0, r_target=0.055):
    """
    Residuals: for each pair and each boundary, penalize if distance < r_target.
    Uses a soft threshold to make it differentiable.
    """
    x = x_flat.reshape(N, 2)
    x = _project_inside(x)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_dists = dists[pair_idx]
    
    # Boundary distances
    rad = np.linalg.norm(x, axis=1)
    bound_dists = 1.0 - rad
    
    # Residuals: penalize when distance < r_target
    # Use smooth min approximation: max(0, r_target - d) approximated smoothly
    # Actually, we want to maximize the minimum, so we penalize violations
    pair_resid = r_target - pair_dists  # positive when violated
    bound_resid = r_target - bound_dists  # positive when violated (distance to boundary < r_target)
    
    # Scale: emphasize tight constraints more
    # Use a soft weighting: weight = 1 / (1 + k * |resid|)
    pair_weight = 1.0 / (1.0 + k * np.abs(pair_resid))
    bound_weight = 1.0 / (1.0 + k * np.abs(bound_resid))
    
    all_resid = np.concatenate([pair_resid * pair_weight, bound_resid * bound_weight])
    
    return all_resid

def _ls_residuals_soft_min(x_flat, pair_idx, k=80.0):
    """
    Alternative: single soft-min residual for the minimum constraint.
    """
    x = x_flat.reshape(N, 2)
    x = _project_inside(x)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(x, axis=1)
    bound_dists = 1.0 - rad
    
    pair_dists = dists[pair_idx]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        soft_min_val = d_min
    else:
        soft_min_val = d_min - (1.0 / k) * np.log(sum_exp)
    
    return np.array([soft_min_val])

def _hex_init():
    """Generate hexagonal lattice initialization."""
    # Optimal hex packing density: each circle has area pi*r^2, 
    # N circles in unit disk: N*pi*r^2 <= pi => r <= 1/sqrt(N)
    # Hex lattice spacing s = 2*r for touching circles
    # Density of hex lattice: (pi*r^2) / (s^2 * sqrt(3)/2) = pi*r^2 / (4*r^2 * sqrt(3)/2) = pi/(2*sqrt(3))
    # So N * (s^2 * sqrt(3)/2) <= pi => s^2 <= 2*pi/(N*sqrt(3))
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    # Use slightly smaller spacing for safety
    s_gen = s_init * 0.92
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 3
    max_i = int(1.0 / s_gen) + 3
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0.0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.98:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        # Keep points closest to center
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            p = np.random.uniform(-0.6, 0.6, 2)
            points = np.vstack([points, [p]])
        points = points[:N]
        
    return points

def _local_refine(points, time_budget, pair_idx, k=60.0):
    """
    Local refinement using least_squares with per-pair residuals.
    """
    start_time = time.time()
    
    # Get current score to set r_target
    current_score = compute_score(points)
    r_target = current_score * 1.01  # Slightly above current
    
    def residuals(x_flat):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            # Return large residuals to signal timeout
            return np.full(len(pair_idx) + N, 100.0)
        return _ls_residuals_per_pair(x_flat, pair_idx, k=k, r_target=r_target)
    
    x0 = points.flatten()
    
    try:
        result = least_squares(
            residuals,
            x0,
            method='trf',
            max_nfev=5000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10
        )
        
        x_opt = result.x.reshape(N, 2)
        x_opt = _project_inside(x_opt)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception:
        return points, compute_score(points)

def _soft_min_refine(points, time_budget, pair_idx, k=80.0):
    """
    Refinement using soft-min objective with L-BFGS-B style via least_squares.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return np.array([100.0])
        return _ls_residuals_soft_min(x_flat, pair_idx, k=k)
    
    x0 = points.flatten()
    
    try:
        result = least_squares(
            residuals,
            x0,
            method='trf',
            max_nfev=3000,
            xtol=1e-10,
            ftol=1e-10
        )
        
        x_opt = result.x.reshape(N, 2)
        x_opt = _project_inside(x_opt)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception:
        return points, compute_score(points)

def _perturb(points, scale=0.02):
    """Perturb points and project inside."""
    p = points + np.random.normal(0, scale, size=points.shape)
    p = _project_inside(p)
    return p

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin (150s limit, keep 32s buffer)
    
    pair_idx = get_pair_indices()
    
    # 1. Initialization
    points = _hex_init()
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Iterative optimization loop
    # Alternate between per-pair LS and soft-min LS with perturbations
    max_iter = 8
    elapsed = time.time() - start_time
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Choose which refinement method to use
        if it % 2 == 0:
            # Per-pair residual method
            time_budget = min(15.0, time_limit - (time.time() - start_time) - 5.0)
            if time_budget < 2.0:
                break
            polished, score = _local_refine(best_points, time_budget, pair_idx, k=60.0)
        else:
            # Soft-min method
            time_budget = min(12.0, time_limit - (time.time() - start_time) - 5.0)
            if time_budget < 2.0:
                break
            polished, score = _soft_min_refine(best_points, time_budget, pair_idx, k=80.0)
        
        if score > best_score:
            best_score = score
            best_points = polished.copy()
        else:
            # Perturb and try again
            if it < max_iter - 1:
                best_points = _perturb(best_points, scale=0.015)
    
    # 3. Final polish with tighter settings
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        final_polished, final_score = _local_refine(best_points, remaining - 2.0, pair_idx, k=100.0)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished
    
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
