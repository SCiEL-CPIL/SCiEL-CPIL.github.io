# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius_for_centres(centres):
    """Compute the maximum common radius for given centres (exact)."""
    n = len(centres)
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)

def _smooth_objective(centres, beta):
    """
    Smooth surrogate: maximize soft-min of all constraints.
    Constraints: pairwise distances / 2, and wall distances.
    Returns (value, gradient) for minimization of -soft_min.
    Vectorized gradient computation.
    """
    n = len(centres)
    # Pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    
    # All constraint values
    # Pair constraints: dists[i,j]/2 for i<j
    # Wall constraints: wall_dists[i,k]
    
    # Extract upper triangle for pairs
    iu = np.triu_indices(n, k=1)
    pair_flat = dists[iu] / 2.0
    
    all_vals = np.concatenate([pair_flat, wall_dists.ravel()])
    
    # Soft min
    beta_safe = max(beta, 1e-6)
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta_safe * (all_vals - min_val))
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    soft_min_val = min_val - log_sum / beta_safe
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(centres)
    
    # Pair contributions (Vectorized)
    # For each pair (i,j), weight w_idx
    # d(soft_min)/d(c_i) += w * (c_i - c_j) / dist_ij
    # d(soft_min)/d(c_j) += w * (c_j - c_i) / dist_ij
    
    # We need to accumulate gradients for each point.
    # diffs[i,j] = c_i - c_j
    # dists[i,j] = norm(diffs[i,j])
    # weights_pair = weights[:len(iu[0])] reshaped to (n, n) upper triangle
    
    weights_pair = np.zeros((n, n))
    weights_pair[iu] = weights[:len(iu[0])]
    
    # Normalize diffs by dists
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_diffs = diffs / safe_dists[:, :, np.newaxis]
    
    # Zero out diagonal and lower triangle contributions for weights
    # We only want upper triangle weights to contribute to lower triangle diffs?
    # Actually, the gradient for c_i from pair (i,j) is w * (c_i - c_j)/dist.
    # The gradient for c_j from pair (i,j) is w * (c_j - c_i)/dist = -w * (c_i - c_j)/dist.
    
    # Sum over j for each i: grad[i] = sum_j (w_ij * (c_i - c_j)/dist_ij + w_ji * (c_i - c_j)/dist_ji)
    # Since w is symmetric (weights are from the same set of values, but assigned to pairs),
    # and the set of pairs is i<j, we have w_ij for i<j.
    # For c_i, contributions come from pairs (i,j) where i<j (as first index) and (j,i) where j<i (as second index).
    # In our `weights_pair` matrix, we only filled upper triangle.
    # Let's make it symmetric for the calculation, or handle it carefully.
    
    # Let's use a symmetric weight matrix W where W[i,j] = weight of pair (i,j) if i!=j.
    W = weights_pair + weights_pair.T
    
    # grad[i] = sum_j W[i,j] * (c_i - c_j) / dist_ij
    # This is equivalent to:
    # grad = sum_j (W * unit_diffs) summed over j
    
    # unit_diffs[i,j,k] = (c_i - c_j)_k / dist_ij
    # We want to sum over j for each i and k.
    # grad[i,k] = sum_j W[i,j] * unit_diffs[i,j,k]
    
    # Matrix multiplication: (W @ unit_diffs) doesn't work directly because unit_diffs is 3D.
    # We can reshape unit_diffs to (n*n, 3) and W to (n*n, n*n)? No.
    
    # Let's do it component-wise or use einsum.
    # grad = np.einsum('ij,ijk->ik', W, unit_diffs)
    
    # Check: W[i,j] is the weight. unit_diffs[i,j,k] is the vector component.
    # Sum over j.
    pair_grad = np.einsum('ij,ijk->ik', W, unit_diffs)
    
    # Wall contributions (Vectorized)
    # wall_dists[i,k] = 0.5 - |c_ik|
    # d(wall_dists[i,k])/d(c_ik) = -sign(c_ik)
    # weights_wall = weights[len(iu[0]):] reshaped to (n, 3)
    
    weights_wall = weights[len(iu[0]):].reshape(n, 3)
    signs = np.sign(centres)
    # If centre is 0, sign is 0, which is a valid subgradient choice (or we can use 0)
    wall_grad = -weights_wall * signs
    
    # Total gradient of soft_min
    grad = pair_grad + wall_grad
    
    # We want to maximize soft_min, so minimize -soft_min
    # Gradient of -soft_min is -grad
    return -soft_min_val, -grad

def _optimize_continuation(start, deadline, max_beta=1500.0):
    """
    Continuation method: ramp up beta from small to large.
    At each beta, run L-BFGS-B to minimize -soft_min.
    """
    centres = start.copy()
    n = len(centres)
    
    # Beta schedule: geometric progression, extended range
    betas = [1.0, 5.0, 15.0, 40.0, 80.0, 150.0, 300.0, 600.0, max_beta]
    
    bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (n * 3)
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def objective_flat(x):
            c = x.reshape(n, 3)
            val, grad = _smooth_objective(c, beta)
            return val, grad.ravel()
        
        x0 = centres.ravel()
        
        try:
            result = minimize(
                objective_flat,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            centres = result.x.reshape(n, 3)
        except Exception:
            pass
        
        # Clip to valid region
        centres = np.clip(centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centres, _radius_for_centres(centres)

def _initial_grid(n_spheres):
    """Create an initial grid configuration."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres]

def _random_perturb(centres, sigma=0.02):
    """Randomly perturb centres."""
    perturbed = centres + np.random.randn(*centres.shape) * sigma
    return np.clip(perturbed, -0.49, 0.49)

def construct():
    """
    Construct 50 sphere centres in the unit cube [-1/2, 1/2]^3 to maximize
    the common radius.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Start with grid
    best_centres = _initial_grid(N)
    best_r = _radius_for_centres(best_centres)
    
    # Continuation optimization from grid
    opt_centres, opt_r = _optimize_continuation(best_centres, deadline)
    if opt_r > best_r:
        best_r = opt_r
        best_centres = opt_centres
    
    # Multiple random restarts with perturbation
    # Dynamic number of restarts based on time
    n_restarts = 20  # Increased due to vectorized gradient speedup
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate perturbed start
        if restart < 3:
            sigma = 0.03
        elif restart < 8:
            sigma = 0.06
        else:
            sigma = 0.1
        
        start = _random_perturb(best_centres, sigma=sigma)
        
        # Optimize with continuation
        opt_centres, opt_r = _optimize_continuation(start, deadline)
        
        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres
    
    # Final high-precision polish with very large beta
    if time.time() < deadline - 5:
        def objective_flat_high_beta(x):
            c = x.reshape(N, 3)
            val, grad = _smooth_objective(c, 3000.0)
            return val, grad.ravel()
        
        bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (N * 3)
        x0 = best_centres.ravel()
        
        try:
            result = minimize(
                objective_flat_high_beta,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            final_centres = result.x.reshape(N, 3)
            final_r = _radius_for_centres(final_centres)
            if final_r > best_r:
                best_r = final_r
                best_centres = final_centres
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
