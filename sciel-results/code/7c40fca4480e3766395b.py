"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_radius(centers, temperature=0.01):
    """Smooth surrogate for the radius using log-sum-exp soft-min."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Soft-min of pairwise distances / 2
    # Use log-sum-exp trick: soft_min(x) = -log(sum(exp(-x/T)))/log(n) approximately
    # Actually: soft_min = -T * log(sum(exp(-x/T)))
    # For stability, subtract max first
    x_pair = sq_dists / 4.0  # (d/2)^2, we want soft min of d/2
    # Work with squared distances for efficiency, then take sqrt at end
    # soft_min of d/2 where d = sqrt(sq_d)
    # Let's use: soft_min(d/2) ≈ sqrt(soft_min(d^2/4))
    
    # Pairwise: soft min of distances
    max_sq = np.max(sq_dists[np.isfinite(sq_dists)])
    exp_terms = np.exp(-(sq_dists - max_sq) / (4.0 * temperature**2))
    np.fill_diagonal(exp_terms, 0.0)
    sum_exp = np.sum(exp_terms)
    if sum_exp > 0:
        soft_min_sq_pair = max_sq - 4.0 * temperature**2 * np.log(sum_exp)
    else:
        soft_min_sq_pair = max_sq
    soft_min_pair = np.sqrt(max(soft_min_sq_pair, 0.0))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / temperature**2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall_sq = max_wall**2 - 2.0 * max_wall * temperature**2 * np.log(sum_wall) / max_wall if max_wall > 0 else 0
        # Simpler: soft_min of wall_dists directly
        soft_min_wall = max_wall - temperature**2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combine: soft min of (soft_min_pair, soft_min_wall)
    # Use another soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / temperature**2)
    soft_min_combined = max_v - temperature**2 * np.log(np.sum(exp_v))
    
    return float(soft_min_combined)

def _objective_and_grad(x, temperature=0.01):
    """Objective: maximize soft-min radius. Returns (-radius, gradient)."""
    centers = x.reshape(N, 3)
    
    # Compute soft-min radius
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    T2 = temperature**2
    
    # Pairwise soft-min
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        exp_terms = np.zeros_like(sq_dists)
        exp_terms[finite_mask] = np.exp(-(sq_dists[finite_mask] - max_sq) / (4.0 * T2))
        sum_exp = np.sum(exp_terms)
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq_pair = max_sq - 4.0 * T2 * log_sum
        else:
            soft_min_sq_pair = max_sq
    else:
        soft_min_sq_pair = 0.0
    
    soft_min_sq_pair = max(soft_min_sq_pair, 0.0)
    soft_min_pair = np.sqrt(soft_min_sq_pair)
    
    # Wall soft-min
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combined soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    if np.any(finite_mask) and sum_exp > 0:
        # d(soft_min_sq_pair)/d(centers[i]) 
        # soft_min_sq_pair = max_sq - 4*T2*log(sum_exp)
        # sum_exp = sum_j exp(-(d_ij^2 - max_sq)/(4*T2))
        # d(sum_exp)/d(x_i) = sum_j exp(...) * (-1/(2*T2)) * d(d_ij^2)/d(x_i)
        # d(d_ij^2)/d(x_i) = 2*(x_i - x_j) for j != i
        # = 2 * sum_{j!=i} (x_i - x_j) * w_j where w_j = exp(...) / sum_exp
        
        # Weighted sum of (x_i - x_j) * w_j
        weights = exp_terms / sum_exp  # (N, N) matrix
        # For each i: sum_j weights[i,j] * (x_i - x_j)
        # = x_i * sum_j weights[i,j] - sum_j weights[i,j] * x_j
        sum_w = np.sum(weights, axis=1)  # (N,)
        weighted_sum_x = weights @ centers  # (N, 3)
        
        # d(soft_min_sq_pair)/d(x_i) = -4*T2 * (1/sum_exp) * sum_j exp_ij * (-1/(2*T2)) * 2*(x_i - x_j)
        # = -4*T2 * (1/sum_exp) * (-1/T2) * sum_j exp_ij * (x_i - x_j)
        # = 4 * (1/sum_exp) * sum_j exp_ij * (x_i - x_j)
        # = 4 * (x_i * sum_j w_ij - sum_j w_ij * x_j)
        
        grad_pair_sq = 4.0 * (centers * sum_w[:, np.newaxis] - weighted_sum_x)
        
        # d(soft_min_pair)/d(x_i) = (1/(2*sqrt(soft_min_sq_pair))) * grad_pair_sq
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        # Combine with wall gradient
        # d(soft_min_combined)/d(x_i) = exp_pair/T2 * grad_pair + exp_wall/T2 * grad_wall / sum_v
        # where exp_pair = exp(-(soft_min_pair - max_v)/T2), etc.
        
        # Wall gradient
        # d(soft_min_wall)/d(x_ik)
        # soft_min_wall = max_wall - T2*log(sum_wall)
        # sum_wall = sum_k exp(-(w_k - max_wall)/T2) where w_k = 0.5 - |x_ik|
        # d(sum_wall)/d(x_ik) = exp(...) * (-1/T2) * d(w_k)/d(x_ik)
        # d(w_k)/d(x_ik) = -sign(x_ik)
        # So d(sum_wall)/d(x_ik) = exp(...) * sign(x_ik) / T2
        
        signs = np.sign(centers)
        signs[centers == 0] = 1.0  # arbitrary at 0
        
        d_sum_wall_dx = exp_wall * signs / T2  # (N, 3)
        d_soft_min_wall_dx = -T2 * d_sum_wall_dx / sum_wall
        
        # Combine
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        
        grad = grad_combined
    
    # We want to MAXIMIZE soft_min_combined, so minimize -soft_min_combined
    return -soft_min_combined, -grad

def _optimize_from_init(init_centers, temperature=0.01, max_iter=5000, time_limit=20.0):
    """Optimize from initial centers using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _objective_and_grad(x, temperature)
        return val, grad
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    return result.x.reshape(N, 3)

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 130.0  # 20 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Strategy: multiple restarts with decreasing temperature
    # Start with grid, then random inits
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Coarse optimization with larger temperature
    temps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    inits = [grid]
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))
    
    for temp in temps:
        if time.time() > deadline - 30.0:
            break
        
        for init in inits:
            if time.time() > deadline - 20.0:
                break
            
            try:
                optimized = _optimize_from_init(init, temperature=temp, 
                                                max_iter=3000, time_limit=15.0)
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update inits with best found so far
        inits = [best_centers] + inits[:3]
    
    # Stage 2: Fine polishing from best with very small temperature
    if time.time() < deadline - 10.0:
        for temp in [0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 5.0:
                break
            
            try:
                optimized = _optimize_from_init(best_centers, temperature=temp,
                                                max_iter=5000, time_limit=10.0)
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
