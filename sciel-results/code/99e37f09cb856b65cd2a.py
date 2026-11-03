"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

def _compute_score(centers):
    """Compute the radius score for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def _softmin_objective_and_grad(x, sigma):
    """Compute objective and analytic gradient for softmin of constraints."""
    centers = x.reshape(N, 3)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise constraints: dist/2 for all pairs (i,j), i != j
    # We'll use the full matrix but only upper triangle for gradient computation
    pair_c = dists / 2.0  # shape (N, N)
    
    # Boundary constraints
    boundary_c = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Stack all finite constraints
    pair_flat = pair_c[np.isfinite(pair_c)]
    bound_flat = boundary_c.ravel()
    all_c = np.concatenate([pair_flat, bound_flat])
    
    # Softmin computation
    scaled = -all_c / sigma
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    softmin_val = -sigma * (max_scaled + np.log(sum_exp))
    objective = -softmin_val  # We minimize -softmin
    
    # Gradient computation
    # f(x) = -softmin(c(x)) = sigma * log(sum(exp(-c_j/sigma)))
    # df/dx = -sum_j weights_j * dc_j/dx
    
    grad = np.zeros_like(x)
    
    # Pairwise gradient contributions
    # For pair (i,j): c_ij = |x_i - x_j|/2
    # dc_ij/dx_i = (x_i - x_j) / (2 * |x_i - x_j|) = (x_i - x_j) / (2 * dist_ij)
    # dc_ij/dx_j = (x_j - x_i) / (2 * |x_i - x_j|) = -(x_i - x_j) / (2 * dist_ij)
    
    # We need to sum over all pairs with their weights
    # The pair_flat array corresponds to the upper triangle of pair_c (since lower triangle has same values)
    # Let's compute using the full matrix and handle symmetry
    
    # Get indices for finite pairs
    finite_mask = np.isfinite(pair_c)
    finite_mask[~np.eye(N, dtype=bool)] = finite_mask[~np.eye(N, dtype=bool)]
    
    # Extract weights for pairs
    # pair_flat contains all finite pair values in row-major order
    # We need to map back to (i,j) indices
    
    # Simpler approach: compute gradient directly from the full matrix
    # For each pair (i,j) with i != j, contribution to grad[i] += weight_ij * (x_i - x_j) / (2 * dist_ij)
    # and grad[j] -= weight_ij * (x_i - x_j) / (2 * dist_ij)
    
    # Get the weights for each pair
    # pair_flat is in the same order as np.where(finite_mask)[0]
    pair_indices = np.where(finite_mask)
    pair_weights = weights[:len(pair_flat)]
    
    # Compute unit vectors
    valid_pairs = pair_indices[1] > pair_indices[0]  # Only upper triangle to avoid double counting
    if np.any(valid_pairs):
        i_idx = pair_indices[0][valid_pairs]
        j_idx = pair_indices[1][valid_pairs]
        w = pair_weights[valid_pairs]
        
        # Direction vectors
        dirs = centers[i_idx] - centers[j_idx]
        dists_valid = dists[i_idx, j_idx]
        
        # Avoid division by zero
        valid_dist = dists_valid > 1e-12
        if np.any(valid_dist):
            dirs = dirs[valid_dist]
            w = w[valid_dist]
            dists_valid = dists_valid[valid_dist]
            i_idx = i_idx[valid_dist]
            j_idx = j_idx[valid_dist]
            
            unit_dirs = dirs / (2.0 * dists_valid[:, np.newaxis])
            
            # Accumulate gradient using add.at
            for k in range(len(i_idx)):
                grad[i_idx[k]] += w[k] * unit_dirs[k]
                grad[j_idx[k]] -= w[k] * unit_dirs[k]
    
    # Boundary gradient contributions
    # c = 0.5 - |x_ik|, dc/dx_ik = -sign(x_ik)
    bound_weights = weights[len(pair_flat):]
    bound_grad = -np.sign(centers) * bound_weights.reshape(N, 3)
    grad += bound_grad
    
    return objective, grad

def _make_bcc_initial():
    """Generate BCC lattice initialization for 50 spheres."""
    # BCC lattice: corners + body center
    # In a unit cube [-0.5, 0.5]^3
    # Use a 3x3x3 supercell of BCC
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append((i/3 - 0.5, j/3 - 0.5, k/3 - 0.5))
                # Body center
                points.append((i/3 - 0.5 + 1/6, j/3 - 0.5 + 1/6, k/3 - 0.5 + 1/6))
    points = np.array(points)
    # Keep 50 closest to center or randomly select
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(points))
    return points[perm[:N]].copy()

def construct():
    start_time = time.time()
    deadline = 140.0  # seconds, leaving 10s margin for safety
    
    best_score = -1
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = []
    
    # Config 1: BCC lattice
    configs.append(_make_bcc_initial())
    
    # Config 2: 4x4x4 grid subset
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    perm = rng.permutation(64)
    kept = perm[:50]
    configs.append(grid[kept].copy())
    
    # Config 3: Random
    configs.append(rng.uniform(-0.42, 0.42, (N, 3)))
    
    # Config 4: Perturbed BCC
    bcc = _make_bcc_initial()
    configs.append(np.clip(bcc + rng.normal(0, 0.03, (N, 3)), -0.48, 0.48))
    
    # Config 5: Random with different seed
    rng2 = np.random.default_rng(123)
    configs.append(rng2.uniform(-0.42, 0.42, (N, 3)))
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for config in configs:
        if time.time() - start_time > deadline - 10:
            break
        
        x = config.reshape(-1).copy()
        
        # Multi-phase annealing with decreasing sigma
        sigmas = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002]
        
        for sigma in sigmas:
            if time.time() - start_time > deadline - 5:
                break
            
            try:
                result = minimize(
                    lambda x, s=sigma: _softmin_objective_and_grad(x, s)[0],
                    x,
                    jac=lambda x, s=sigma: _softmin_objective_and_grad(x, s)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                x = result.x
            except Exception:
                pass
        
        # Final polish with very small sigma
        if time.time() - start_time < deadline - 3:
            try:
                result = minimize(
                    lambda x: _softmin_objective_and_grad(x, 0.001)[0],
                    x,
                    jac=lambda x: _softmin_objective_and_grad(x, 0.001)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
                )
                x = result.x
            except Exception:
                pass
        
        centers = x.reshape(N, 3)
        centers = np.clip(centers, -0.5, 0.5)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Fallback
    if best_centers is None or best_score < 0.12:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

# EVOLVE-BLOCK-END
