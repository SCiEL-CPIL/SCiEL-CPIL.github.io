import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_obj_and_grad(x, beta=50.0):
    """
    Soft-min objective: maximize the soft minimum of all pairwise distances/2 and wall distances.
    Using log-sum-exp for numerical stability.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    pair_dists = np.sqrt(dists_sq) / 2.0
    
    # Wall distances
    abs_centers = np.abs(centers)
    wall_dists = HALF - np.max(abs_centers, axis=1)
    
    # Combine all constraints: pair_dists (upper triangle) and wall_dists
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals = pair_dists[i_idx, j_idx]
    all_vals = np.concatenate([pair_vals, wall_dists])
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * v))) / beta
    # For numerical stability, shift by max
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_val - log_sum / beta
    
    # Gradient: d(soft_min)/d(all_vals) = exp(-beta*(v - max_val)) / sum(exp(-beta*(v - max_val)))
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient with respect to pair distances
    grad_pair = weights[:len(pair_vals)]  # shape (n_pairs,)
    grad_wall = weights[len(pair_vals):]  # shape (n,)
    
    # Gradient of pair distance w.r.t. centers
    grad = np.zeros_like(centers)
    
    # For each pair (i,j), d(pair_dist)/d(centers[i]) = (centers[i] - centers[j]) / (2 * pair_dist)
    # We need to accumulate: sum over j of grad_pair[k] * (c_i - c_j) / (2 * d_ij)
    
    # Vectorized: for each center i, sum over all j != i of w_{ij} * (c_i - c_j) / d_{ij}
    # where w_{ij} = grad_pair for the pair index
    
    # Build a weight matrix W where W[i,j] = grad_pair for pair (i,j)
    W = np.zeros((n, n))
    W[i_idx, j_idx] = grad_pair
    W[j_idx, i_idx] = grad_pair
    
    # dists matrix (with inf on diagonal)
    dist_mat = np.sqrt(dists_sq)
    np.fill_diagonal(dist_mat, 1.0)  # avoid division by zero
    
    # grad[i] = sum_j W[i,j] * (c_i - c_j) / dist_mat[i,j]
    # = sum_j W[i,j] * c_i / dist_mat[i,j] - sum_j W[i,j] * c_j / dist_mat[i,j]
    
    W_over_d = W / dist_mat  # W[i,j] / d_{ij}
    
    # term1[i] = c_i * sum_j W_over_d[i,j]
    sum_W_over_d = np.sum(W_over_d, axis=1)
    term1 = centers * sum_W_over_d[:, np.newaxis]
    
    # term2[i] = sum_j W_over_d[i,j] * c_j
    term2 = W_over_d @ centers
    
    grad_pair_part = term1 - term2
    
    # Gradient of wall distance: d(wall_dist)/d(c_i) = -sign(c_i) for the dominant axis
    # wall_dist_i = 0.5 - max_k |c_ik|
    # d(wall_dist_i)/d(c_ik) = -sign(c_ik) if |c_ik| = max, else 0
    
    max_abs = np.max(abs_centers, axis=1)
    # For ties, distribute gradient equally (or just pick one; let's pick all max axes)
    for i in range(n):
        max_val_i = max_abs[i]
        mask = np.abs(abs_centers[i] - max_val_i) < 1e-12
        if np.any(mask):
            grad_wall_part_i = -np.sign(centers[i]) * mask / np.sum(mask)
            grad[i] += grad_wall[i] * grad_wall_part_i
    
    grad += grad_pair_part
    
    return -soft_min, grad.flatten()  # minimize negative soft_min

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = HALF - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        # Vectorized gradient for wall constraints
        # For each violating center, gradient is -2*v*sign(c) for the dominant axis
        max_vals = max_abs[i_valid_wall]
        abs_vals = abs_centers[i_valid_wall]
        
        # Find dominant axes (handle ties by distributing)
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_val = max_abs[idx]
            # Find all axes that achieve the max
            mask_max = np.abs(abs_centers[idx] - max_val) < 1e-12
            if np.any(mask_max):
                grad[idx] += -2.0 * v * np.sign(centers[idx]) * mask_max / np.sum(mask_max)
            
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _minimize_soft_min(init_centers, beta=50.0, max_iter=300, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the negative soft-min objective to maximize the minimum distance.
    """
    def objective_and_grad(x):
        val, g = _soft_min_obj_and_grad(x, beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers, res.success
    except Exception:
        return init_centers, False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: 5x4x5 Grid with symmetry breaking
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    best_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Phase 1: Optimize with soft-min surrogate to get a good starting point
    soft_pos, _ = _minimize_soft_min(best_pos, beta=20.0, max_iter=200)
    soft_r = _compute_radius(soft_pos)
    
    # Also try with higher beta for sharper min
    soft_pos2, _ = _minimize_soft_min(soft_pos, beta=50.0, max_iter=200)
    soft_r2 = _compute_radius(soft_pos2)
    
    if soft_r2 > soft_r:
        best_pos = soft_pos2
        best_r = soft_r2
    else:
        best_r = soft_r
    
    # Phase 2: Iterative radius increase with energy minimization
    current_r = best_r
    step_size = 0.0003
    consecutive_failures = 0
    max_failures = 4
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-8:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.003, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.01, size=(N, 3)))
        
        # Occasionally try a soft-min restart to escape local minima
        if consecutive_failures >= 2 and time.time() < deadline - 30.0:
            soft_restart, _ = _minimize_soft_min(best_pos + rng_search.normal(0, 0.05, size=(N, 3)), beta=30.0, max_iter=100)
            candidates.append(soft_restart)
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            max_iter = 400 if current_r < 0.11 else 600
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            actual_r = _compute_radius(new_pos)
            
            if actual_r >= target_r - 1e-7:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
                break
            elif actual_r > current_r:
                # Accept improvement even if target not fully met
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                # Don't count as full success for step size logic
        
        if not success:
            consecutive_failures += 1
            
            # Polish current best
            if time.time() < deadline - 25.0 and consecutive_failures % 3 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=300)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        # Try to push slightly beyond current_r
        for push in [1e-5, 5e-6, 2e-6]:
            if time.time() > deadline - 3.0:
                break
            final_target = best_r + push
            polished_pos, final_energy, _ = _minimize_energy(best_pos, final_target, max_iter=1500, ftol=1e-15, gtol=1e-14)
            final_r = _compute_radius(polished_pos)
            
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r
        
        # One more polish at the exact best_r
        if time.time() < deadline - 1.0:
            polished_pos, _, _ = _minimize_energy(best_pos, best_r, max_iter=800, ftol=1e-15, gtol=1e-14)
            final_r = _compute_radius(polished_pos)
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r

    return best_pos
