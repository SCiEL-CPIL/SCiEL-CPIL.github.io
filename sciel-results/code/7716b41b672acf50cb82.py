"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(centers):
    """Compute the radius score for given centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary distances
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def _softmin_objective(x, sigma):
    """Objective: minimize -softmin(constraints). x is flattened centers (N*3,)."""
    centers = x.reshape(N, 3)
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    # Boundary constraints
    boundary = 0.5 - np.abs(centers)
    boundary_constraints = boundary.reshape(-1)
    # All constraints
    all_c = np.concatenate([pair_constraints.ravel(), boundary_constraints])
    # Filter out inf
    all_c = all_c[np.isfinite(all_c)]
    # Softmin
    # -sigma * log(sum(exp(-c/sigma)))
    # To avoid overflow, subtract max
    scaled = -all_c / sigma
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    softmin_val = -sigma * log_sum
    return -softmin_val  # We minimize the negative

def _softmin_gradient(x, sigma):
    """Gradient of the objective."""
    centers = x.reshape(N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    boundary = 0.5 - np.abs(centers)
    boundary_constraints = boundary.reshape(-1)
    all_c = np.concatenate([pair_constraints.ravel(), boundary_constraints])
    all_c = all_c[np.isfinite(all_c)]
    scaled = -all_c / sigma
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of -softmin = gradient of sigma * log(sum(exp(-c/sigma)))
    # d/dc_i of sigma * log(sum(exp(-c_j/sigma))) = sigma * (1/sum) * exp(-c_i/sigma) * (-1/sigma) = -exp(-c_i/sigma)/sum = -weights[i]
    # So gradient of objective w.r.t. c_i = -weights[i]
    # But we need gradient w.r.t. x (the centers)
    
    # This is getting complex. Let me just use numerical gradient or let L-BFGS compute it.
    # Actually, let me just return None and let scipy use numerical gradient, or compute analytically.
    
    # For simplicity, let me compute the gradient numerically with a small step.
    # But that's slow. Let me think...
    
    # Actually, let me just use the analytic gradient. 
    # The objective is f(x) = -softmin(c(x)) where c(x) are the constraints.
    # softmin(c) = -sigma * log(sum(exp(-c_i/sigma)))
    # f(x) = sigma * log(sum(exp(-c_i/sigma)))
    # df/dx = sigma * (1/sum) * sum_j exp(-c_j/sigma) * (-1/sigma) * dc_j/dx
    #        = -sum_j weights_j * dc_j/dx
    # where weights_j = exp(-c_j/sigma) / sum(exp(-c_k/sigma))
    
    # So I need dc_j/dx for each constraint j.
    
    # For pairwise constraint j (pair (i,k)): c_j = |x_i - x_k| / 2
    # dc_j/dx_i = (x_i - x_k) / (2 * |x_i - x_k|) = (x_i - x_k) / (2 * dist_ik)
    # dc_j/dx_k = (x_k - x_i) / (2 * |x_i - x_k|) = -(x_i - x_k) / (2 * dist_ik)
    
    # For boundary constraint: c = 0.5 - |x_ik|
    # dc/dx_ik = -sign(x_ik)
    
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    # all_c[0:1225] are pairwise, all_c[1225:] are boundary
    n_pairs = N * (N - 1) // 2
    pair_weights = weights[:n_pairs]
    boundary_weights = weights[n_pairs:]
    
    # Pairwise: need to map flat index to (i, k) pair
    # The pairwise constraints were computed from the upper triangle of the distance matrix
    # Let me recompute the mapping
    # Actually, I flattened the distance matrix which includes both (i,k) and (k,i)
    # That's a problem. Let me fix the constraint construction.
    
    # Let me redo this more carefully.
    # I'll only use unique pairs (i < k) for pairwise constraints.
    
    # Let me restructure.
    return None  # Fall back to numerical gradient for now

def construct():
    start_time = time.time()
    deadline = 130.0  # seconds, leaving 20s margin
    
    best_score = -1
    best_centers = None
    
    # Initial configurations to try
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configs
    configs = []
    
    # Config 1: Random
    configs.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    # Config 2: 4x4x4 grid with 14 random positions removed
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Remove 14 random positions
    perm = rng.permutation(64)
    kept = perm[:50]
    configs.append(grid[kept].copy())
    
    # Config 3: Another random
    configs.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    # Config 4: Perturbed grid
    grid_perturbed = grid[kept].copy() + rng.normal(0, 0.05, (50, 3))
    grid_perturbed = np.clip(grid_perturbed, -0.48, 0.48)
    configs.append(grid_perturbed)
    
    # Config 5: Another random with different seed
    rng2 = np.random.default_rng(123)
    configs.append(rng2.uniform(-0.45, 0.45, (N, 3)))
    
    # Config 6: Face-centered cubic-like
    # FCC lattice points in the cube
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fcc_points.append((i/3 - 0.5, j/3 - 0.5, k/3 - 0.5))
                fcc_points.append((i/3 - 0.5, j/3 - 0.5 + 1/6, k/3 - 0.5 + 1/6))
    fcc_points = np.array(fcc_points)
    # Keep only 50
    perm = rng.permutation(len(fcc_points))
    configs.append(fcc_points[perm[:50]].copy())
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for config in configs:
        if time.time() - start_time > deadline - 5:
            break
        
        x = config.reshape(-1)
        
        # Multi-phase annealing
        sigmas = [0.05, 0.01, 0.003]
        
        for sigma in sigmas:
            if time.time() - start_time > deadline - 5:
                break
            
            # Use numerical gradient (slow but reliable)
            # Actually, let me try to compute analytic gradient properly
            # For now, use BFGS with numerical gradient
            try:
                result = minimize(
                    _softmin_objective,
                    x,
                    args=(sigma,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x = result.x
            except Exception:
                pass
        
        # Compute final score
        centers = x.reshape(N, 3)
        # Clip to valid range
        centers = np.clip(centers, -0.5, 0.5)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # If we didn't get a good result, fall back to grid
    if best_centers is None or best_score < 0.12:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

# EVOLVE-BLOCK-END
