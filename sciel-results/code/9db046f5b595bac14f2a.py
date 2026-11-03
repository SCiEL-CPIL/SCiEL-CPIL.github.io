import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    
    # Build a weight matrix W where W[i,j] = grad_pair for pair (i,j)
    W = np.zeros((n, n))
    W[i_idx, j_idx] = grad_pair
    W[j_idx, i_idx] = grad_pair
    
    # dists matrix (with inf on diagonal)
    dist_mat = np.sqrt(dists_sq)
    np.fill_diagonal(dist_mat, 1.0)  # avoid division by zero
    
    # grad[i] = sum_j W[i,j] * (c_i - c_j) / dist_mat[i,j]
    W_over_d = W / dist_mat  # W[i,j] / d_{ij}
    
    # term1[i] = c_i * sum_j W_over_d[i,j]
    sum_W_over_d = np.sum(W_over_d, axis=1)
    term1 = centers * sum_W_over_d[:, np.newaxis]
    
    # term2[i] = sum_j W_over_d[i,j] * c_j
    term2 = W_over_d @ centers
    
    grad_pair_part = term1 - term2
    
    # Gradient of wall distance: d(wall_dist)/d(c_i) = -sign(c_i) for the dominant axis
    max_abs = np.max(abs_centers, axis=1)
    # For ties, distribute gradient equally
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
    
    # 1. Pairwise terms - vectorized
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
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

    # 2. Wall terms - fully vectorized
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
        # Create a mask for max axes
        max_mask = np.abs(abs_vals - max_vals[:, np.newaxis]) < 1e-12
        # Sum of masks for normalization
        mask_sums = np.sum(max_mask, axis=1, keepdims=True)
        mask_sums = np.where(mask_sums == 0, 1, mask_sums)
        
        # Gradient contribution: -2 * v * sign(c) * mask / mask_sum
        signs = np.sign(centers[i_valid_wall])
        grad_contrib = -2.0 * violations_wall[:, np.newaxis] * signs * max_mask / mask_sums
        
        np.add.at(grad, i_valid_wall, grad_contrib)
            
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

def _bcc_init():
    """
    Generate BCC (body-centered cubic) lattice points.
    This is a known good packing structure.
    """
    # For 100 spheres, we can use a 5x4x5 grid of BCC unit cells
    # Each BCC unit cell has 2 points: corners and center
    # Actually, let's use a simpler approach: generate a grid and add center points
    
    # Use a 5x4x5 grid of points, then add center points
    nx, ny, nz = 5, 4, 5
    points = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Corner point
                x = -0.5 + (ix + 0.5) / nx
                y = -0.5 + (iy + 0.5) / ny
                z = -0.5 + (iz + 0.5) / nz
                points.append([x, y, z])
                
                # Center point (if not at boundary)
                if ix < nx-1 and iy < ny-1 and iz < nz-1:
                    cx = -0.5 + (ix + 1.0) / nx
                    cy = -0.5 + (iy + 1.0) / ny
                    cz = -0.5 + (iz + 1.0) / nz
                    points.append([cx, cy, cz])
    
    # Take first N points
    points = np.array(points[:N])
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(12345)
    points += rng.normal(0, 0.005, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

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
    
    # 1. Initialization: Try multiple initialization strategies
    # Strategy A: BCC lattice
    bcc_pos = _bcc_init()
    
    # Strategy B: Random initialization
    rng = np.random.default_rng(42)
    random_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
    
    # Strategy C: Grid initialization
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    grid_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    grid_pos = np.clip(grid_pos, -0.49, 0.49)
    
    # Evaluate all initializations with soft-min
    initializations = [bcc_pos, random_pos, grid_pos]
    best_pos = None
    best_r = 0.0
    
    for init_pos in initializations:
        if time.time() > deadline - 60.0:
            break
        
        # Optimize with soft-min surrogate
        soft_pos, _ = _minimize_soft_min(init_pos, beta=20.0, max_iter=150)
        soft_r = _compute_radius(soft_pos)
        
        if soft_r > best_r:
            best_r = soft_r
            best_pos = soft_pos
    
    # Phase 2: Iterative radius increase with energy minimization
    current_r = best_r
    step_size = 0.0002
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 30.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-9:
                break
        
        # Generate diverse candidates
        candidates = [best_pos.copy()]
        
        # Small perturbations
        candidates.append(best_pos + rng_search.normal(0, 0.002, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.005, size=(N, 3)))
        
        # Medium perturbation
        if consecutive_failures >= 2:
            candidates.append(best_pos + rng_search.normal(0, 0.01, size=(N, 3)))
        
        # BCC restart if stuck
        if consecutive_failures >= 4 and time.time() < deadline - 40.0:
            bcc_restart = _bcc_init()
            bcc_soft, _ = _minimize_soft_min(bcc_restart, beta=30.0, max_iter=100)
            candidates.append(bcc_soft)
        
        success = False
        for start_pos in candidates:
            if time.time() > deadline - 35.0:
                break
            
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Adaptive max_iter based on progress
            if current_r < 0.105:
                max_iter = 300
            elif current_r < 0.11:
                max_iter = 400
            else:
                max_iter = 500
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            actual_r = _compute_radius(new_pos)
            
            if actual_r >= target_r - 1e-7:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
                break
            elif actual_r > current_r + 1e-9:
                # Accept improvement even if target not fully met
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
        
        if not success:
            consecutive_failures += 1
            
            # Polish current best periodically
            if time.time() < deadline - 35.0 and consecutive_failures % 3 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=200)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10.0:
        # Try to push slightly beyond current_r with multiple attempts
        for push in [2e-5, 1e-5, 5e-6, 2e-6, 1e-6]:
            if time.time() > deadline - 5.0:
                break
            final_target = best_r + push
            polished_pos, final_energy, _ = _minimize_energy(best_pos, final_target, max_iter=800, ftol=1e-15, gtol=1e-14)
            final_r = _compute_radius(polished_pos)
            
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r
        
        # Final polish at the exact best_r
        if time.time() < deadline - 2.0:
            polished_pos, _, _ = _minimize_energy(best_pos, best_r, max_iter=500, ftol=1e-15, gtol=1e-14)
            final_r = _compute_radius(polished_pos)
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r

    return best_pos
