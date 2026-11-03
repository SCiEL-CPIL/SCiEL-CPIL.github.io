"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_radius_and_grad(x, temperature=0.01):
    """
    Compute soft-min radius surrogate and its gradient.
    x is flat array of length N*3.
    Returns (radius_value, gradient_of_radius).
    """
    centers = x.reshape(N, 3)
    T2 = temperature**2
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    finite_mask = np.isfinite(sq_dists)
    if not np.any(finite_mask):
        return 0.0, np.zeros_like(x)
    
    # Soft-min of pairwise distances/2
    # We want soft_min of d_ij/2 = soft_min of sqrt(sq_dists)/2
    # Use soft-min on sq_dists/4 then sqrt
    x_pair = sq_dists / 4.0  # (d/2)^2
    
    max_x = np.max(x_pair[finite_mask])
    exp_terms = np.zeros_like(x_pair)
    exp_terms[finite_mask] = np.exp(-(x_pair[finite_mask] - max_x) / T2)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp > 0:
        log_sum = np.log(sum_exp)
        soft_min_sq = max_x - T2 * log_sum
        soft_min_sq = max(soft_min_sq, 0.0)
    else:
        soft_min_sq = max_x
    
    soft_min_pair = np.sqrt(soft_min_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combine: soft min of (soft_min_pair, soft_min_wall)
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp  # (N, N)
        sum_w = np.sum(weights, axis=1)  # (N,)
        weighted_sum_x = weights @ centers  # (N, 3)
        
        # d(soft_min_sq)/d(centers_i) = 2 * (centers_i * sum_w_i - sum_j w_ij * centers_j)
        # Actually: soft_min_sq = max_x - T2 * log(sum_exp)
        # d(log(sum_exp))/d(x_pair_ij) = exp_terms_ij / sum_exp = weights_ij
        # d(x_pair_ij)/d(centers_i) = 2 * (centers_i - centers_j)
        # So d(soft_min_sq)/d(centers_i) = -T2 * sum_j weights_ij * 2*(centers_i - centers_j)
        # = -2*T2 * (centers_i * sum_j weights_ij - sum_j weights_ij * centers_j)
        # = -2*T2 * (centers_i * sum_w_i - weighted_sum_x_i)
        
        grad_soft_min_sq = -2.0 * T2 * (centers * sum_w[:, np.newaxis] - weighted_sum_x)
        
        if soft_min_sq > 1e-15:
            # d(soft_min_pair)/d(centers_i) = grad_soft_min_sq / (2 * soft_min_pair)
            grad_pair = grad_soft_min_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
    else:
        grad_pair = np.zeros_like(centers)
    
    # Wall gradient
    signs = np.sign(centers)
    signs[centers == 0] = 1.0
    
    # d(soft_min_wall)/d(centers_i_k) = -T2 * d(log(sum_wall))/d(wall_dists_ik) * d(wall_dists_ik)/d(centers_ik)
    # d(log(sum_wall))/d(wall_dists_ik) = exp_wall_ik / sum_wall
    # d(wall_dists_ik)/d(centers_ik) = -sign(centers_ik)
    # So d(soft_min_wall)/d(centers_ik) = -T2 * (exp_wall_ik / sum_wall) * (-sign(centers_ik))
    # = T2 * exp_wall_ik * sign(centers_ik) / sum_wall
    
    if sum_wall > 0:
        grad_wall = T2 * exp_wall * signs / sum_wall
    else:
        grad_wall = np.zeros_like(centers)
    
    # Combine gradients
    exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
    exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
    
    grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * grad_wall
    
    return float(soft_min_combined), grad_combined.flatten()

def _objective_and_grad(x, temperature=0.01):
    """Objective for minimization: minimize -soft_min_radius."""
    radius, grad = _soft_min_radius_and_grad(x, temperature)
    return -radius, -grad

def _optimize_lbfgsb(init_centers, temperature=0.01, max_iter=5000, time_limit=20.0):
    """Optimize from initial centers using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _objective_and_grad(x, temperature)
        return val, grad
    
    try:
        result = minimize(
            obj, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return init_centers.copy()

def _sgld_step(centers, temperature, lr, noise_scale):
    """
    One SGLD step: update = -lr * grad + noise
    We want to maximize soft_min_radius, so gradient ascent.
    """
    x = centers.flatten()
    radius, grad = _soft_min_radius_and_grad(x, temperature)
    
    noise = np.random.normal(0, noise_scale, size=x.shape)
    x_new = x + lr * grad + noise
    
    centers_new = x_new.reshape(N, 3)
    centers_new = np.clip(centers_new, -0.5, 0.5)
    
    return centers_new, radius

def _sgld_anneal(init_centers, start_temp=0.05, end_temp=0.001, 
                 start_lr=0.001, end_lr=0.00001,
                 start_noise=0.01, end_noise=0.0001,
                 n_steps=2000, time_limit=30.0):
    """
    SGLD with adaptive annealing of temperature, learning rate, and noise.
    """
    start_time = time.time()
    centers = init_centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        progress = step / n_steps
        
        # Anneal parameters
        temp = start_temp * (end_temp / start_temp) ** progress
        lr = start_lr * (end_lr / start_lr) ** progress
        noise = start_noise * (end_noise / start_noise) ** progress
        
        centers_new, radius = _sgld_step(centers, temp, lr, noise)
        
        # Check if improved
        actual_radius = _compute_radius(centers_new)
        if actual_radius > best_radius:
            best_radius = actual_radius
            best_centers = centers_new.copy()
        
        centers = centers_new
    
    return best_centers, best_radius

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        min_sq_dist = np.min(sq_dists)
        min_pair_dist = np.sqrt(min_sq_dist)
        min_pair_idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
        
        min_wall_dist = np.min(0.5 - np.abs(centers))
        min_wall_idx = np.unravel_index(np.argmin(0.5 - np.abs(centers)), (N, 3))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            push = (min_pair_dist / 2.0 - r) * d / dist * 0.5
            if push > 0:
                centers[i] += push
                centers[j] -= push
        else:
            i, k = min_wall_idx
            wall = 0.5 - np.abs(centers[i, k])
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _bcc_init(n):
    """Generate body-centered cubic lattice initial configuration."""
    m = int(np.ceil((n / 2) ** (1.0/3.0)))
    spacing = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = (ix + 0.5) * spacing - 0.5
                y = (iy + 0.5) * spacing - 0.5
                z = (iz + 0.5) * spacing - 0.5
                centers.append([x, y, z])
                if ix < m-1 and iy < m-1 and iz < m-1:
                    bx = (ix + 1.0) * spacing - 0.5
                    by = (iy + 1.0) * spacing - 0.5
                    bz = (iz + 1.0) * spacing - 0.5
                    centers.append([bx, by, bz])
    
    centers = np.array(centers)
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        grid = _grid_init(n)
        centers = np.vstack([centers, grid[len(centers):n]])
    
    return centers

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _jittered_grid_init(n, seed=None):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-0.02, 0.02, size=(n, 3))
    return np.clip(grid + jitter, -0.49, 0.49)

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Generate initializations
    inits = []
    grid = _grid_init(N)
    inits.append(('grid', grid))
    
    bcc = _bcc_init(N)
    inits.append(('bcc', bcc))
    
    for seed in range(8):
        inits.append((f'rand{seed}', _random_init(N, seed=seed)))
    
    for seed in range(6):
        inits.append((f'jitter{seed}', _jittered_grid_init(N, seed=seed)))
    
    # Evaluate all inits
    for name, init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # Stage 1: L-BFGS-B optimization with temperature annealing
    temps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    active_inits = [best_centers]
    for name, init in inits[:6]:
        active_inits.append(init)
    
    for temp in temps:
        if time.time() > deadline - 60.0:
            break
        
        for init in active_inits:
            if time.time() > deadline - 45.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 30.0
                if time_remaining < 5.0:
                    break
                optimized = _optimize_lbfgsb(init, temperature=temp, 
                                             max_iter=3000, time_limit=min(12.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update active inits
        active_inits = [best_centers] + active_inits[:3]
    
    # Stage 2: SGLD annealing from best configuration
    if time.time() < deadline - 40.0:
        try:
            time_remaining = deadline - time.time() - 25.0
            sgld_centers, sgld_radius = _sgld_anneal(
                best_centers, 
                start_temp=0.05, end_temp=0.001,
                start_lr=0.001, end_lr=0.00001,
                start_noise=0.008, end_noise=0.0001,
                n_steps=3000,
                time_limit=min(30.0, time_remaining)
            )
            if sgld_radius > best_radius:
                best_radius = sgld_radius
                best_centers = sgld_centers.copy()
        except Exception:
            pass
    
    # Stage 3: Fine L-BFGS-B polishing
    if time.time() < deadline - 25.0:
        fine_temps = [0.0005, 0.0002, 0.0001, 0.00005]
        for temp in fine_temps:
            if time.time() > deadline - 12.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 8.0
                if time_remaining < 3.0:
                    break
                optimized = _optimize_lbfgsb(best_centers, temperature=temp,
                                             max_iter=5000, time_limit=min(8.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 4: Final critical-pair polish
    if time.time() < deadline - 5.0:
        try:
            polished = _polish_critical_pairs(best_centers, time_limit=4.0)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
