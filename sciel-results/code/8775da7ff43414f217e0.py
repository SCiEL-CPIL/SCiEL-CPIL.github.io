"""Improved packing: multi-stage soft-min annealing with diverse initializations."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the valid radius for a set of centers in [-0.5, 0.5]^3."""
    n = len(centers)
    if n != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Wall distances
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def _soft_min_energy(centers, beta):
    """
    Compute a smooth surrogate for the negative radius.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Equivalently, minimize -r.
    
    Using soft-min: softmin(x) = -1/beta * log(sum(exp(-beta*x)))
    We define energy = softmin of all constraints (pairwise half-distances and wall distances).
    Minimizing energy ≈ maximizing the min constraint.
    
    Returns (energy, grad) where grad is d(energy)/d(centers).
    """
    n = len(centers)
    # Wall constraints: for each center i, wall_dist = 0.5 - max_k |c_ik|
    # Actually wall constraint is 0.5 - |c_ik| for each coordinate.
    # Let's use per-coordinate wall distances: w_ik = 0.5 - |c_ik|
    # And pairwise: d_ij/2 for i<j
    
    # Collect all constraint values
    # Wall: n*3 values
    wall_vals = 0.5 - np.abs(centers)  # shape (n, 3)
    wall_flat = wall_vals.ravel()  # shape (3n,)
    
    # Pairwise: n*(n-1)/2 values
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu]  # shape (n*(n-1)/2,)
    half_dists = pair_vals / 2.0
    
    # Combine all constraints
    all_vals = np.concatenate([wall_flat, half_dists])
    
    # Soft-min with large beta
    # To avoid overflow, shift by max
    v_max = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - v_max))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = v_max - log_sum / beta
    
    # Gradient: d(softmin)/d(all_vals[k]) = -exp_terms[k] / sum(exp_terms)
    # Note: softmin = v_max - (1/beta)*log(sum(exp(-beta*(v - v_max))))
    # d(softmin)/d(v_k) = -1/beta * d/d(v_k) log(sum(exp(-beta*(v_j - v_max))))
    # = -1/beta * sum_j [-beta * exp(-beta*(v_j-v_max)) * delta(j,k) / sum]
    # = exp(-beta*(v_k-v_max)) / sum(exp(...))
    # Wait, let me recompute.
    # f = -1/beta * log(sum_j exp(-beta*v_j + beta*v_max))
    # df/dv_k = -1/beta * (-beta * exp(-beta*v_k + beta*v_max)) / sum_j exp(-beta*v_j + beta*v_max)
    # = exp(-beta*(v_k - v_max)) / sum_j exp(-beta*(v_j - v_max))
    # = exp_terms[k] / sum(exp_terms)
    
    grad_all = exp_terms / np.sum(exp_terms)
    
    # Now back-propagate to centers
    grad = np.zeros_like(centers)
    
    # Wall part: wall_flat[k] = 0.5 - |c_{i,k%3}| where i = k//3
    # d(wall_flat[k])/d(c_{i,k%3}) = -sign(c_{i,k%3})
    for i in range(n):
        for k in range(3):
            idx = i * 3 + k
            g = grad_all[idx]
            c_val = centers[i, k]
            if c_val > 0:
                grad[i, k] -= g
            elif c_val < 0:
                grad[i, k] += g
            else:
                grad[i, k] -= g  # arbitrary, measure zero
    
    # Pairwise part: half_dists[m] = |c_i - c_j| / 2
    # d(half_dist)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2*dist)
    # d(half_dist)/d(c_j) = (c_j - c_i) / (2 * dist)
    for m in range(len(pair_vals)):
        i, j = iu[0][m], iu[1][m]
        g = grad_all[3*n + m]
        d = pair_vals[m]
        if d > 1e-12:
            direction = (centers[i] - centers[j]) / d
            grad[i] += g * direction / 2.0
            grad[j] -= g * direction / 2.0
    
    return softmin_val, grad

def _optimize_stage(centers, betas, iters_per_beta, lr_start, lr_end, time_limit):
    """Run multi-stage soft-min optimization."""
    t0 = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    n_betas = len(betas)
    total_iters = sum(iters_per_beta)
    
    for b_idx, beta in enumerate(betas):
        # Learning rate schedule
        if b_idx == 0:
            lr = lr_start
        else:
            progress = b_idx / max(1, n_betas - 1)
            lr = lr_start * (lr_end / lr_start) ** progress
        
        for it in range(iters_per_beta):
            if time.time() - t0 > time_limit:
                return best_centers, best_radius
            
            energy, grad = _soft_min_energy(centers, beta)
            
            # Gradient ascent on radius = gradient descent on -radius
            # We want to maximize radius, so move in direction of -grad(energy)
            # Actually energy approximates -radius, so maximizing radius = minimizing energy
            # So we do: centers -= lr * grad(energy)
            
            centers_new = centers - lr * grad
            
            # Project back to cube
            centers_new = np.clip(centers_new, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check improvement
            r_new = _compute_radius(centers_new)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = centers_new.copy()
            
            centers = centers_new
    
    return best_centers, best_radius

def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_fcc():
    """FCC-like initialization."""
    centers = []
    # Simple layered approach
    layer_spacing = 0.15
    n_layers = int(1.0 / layer_spacing) + 1
    for l in range(n_layers):
        z = -0.5 + l * layer_spacing
        if z > 0.5:
            break
        # Alternate offset
        offset = 0.0 if l % 2 == 0 else 0.075
        n_per_layer = 5
        for i in range(n_per_layer):
            for j in range(n_per_layer):
                x = -0.5 + (i + 0.5) * 0.2 + offset
                y = -0.5 + (j + 0.5) * 0.2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    centers.append([x, y, z])
        if len(centers) >= N:
            break
    centers = np.array(centers[:N])
    return centers

def _init_random():
    """Random initialization."""
    return (np.random.rand(N, 3) - 0.5) * 0.95

def _init_layered():
    """Layered hexagonal-like initialization."""
    centers = []
    n_layers = 7
    layer_z = np.linspace(-0.45, 0.45, n_layers)
    for l, z in enumerate(layer_z):
        offset = 0.0 if l % 2 == 0 else 0.06
        n_per = 8
        for i in range(n_per):
            for j in range(n_per):
                x = -0.45 + (i + 0.5) * 0.12 + offset
                y = -0.45 + (j + 0.5) * 0.12
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    centers.append([x, y, z])
        if len(centers) >= N:
            break
    centers = np.array(centers[:N])
    return centers

def construct():
    t_start = time.time()
    time_limit = 115.0  # Safety margin
    
    initializers = [_init_grid, _init_fcc, _init_random, _init_layered]
    
    best_centers = None
    best_radius = -1.0
    
    for init_func in initializers:
        if time.time() - t_start > time_limit - 10:
            break
        
        centers = init_func()
        r0 = _compute_radius(centers)
        
        if r0 <= best_radius and best_centers is not None:
            # Skip if not promising
            continue
        
        # Multi-stage annealing
        betas = [10, 50, 200, 1000, 5000, 20000, 100000, 400000]
        iters_per_beta = [200, 200, 200, 200, 150, 100, 50, 30]
        
        centers_opt, r_opt = _optimize_stage(
            centers, betas, iters_per_beta,
            lr_start=0.005, lr_end=0.0001,
            time_limit=time_limit - (time.time() - t_start) - 5
        )
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = centers_opt.copy()
    
    # Final polish: high-precision local optimization
    if best_centers is not None and time.time() - t_start < time_limit - 5:
        polish_betas = [400000, 1000000, 5000000]
        polish_iters = [50, 30, 20]
        best_centers, best_radius = _optimize_stage(
            best_centers, polish_betas, polish_iters,
            lr_start=0.00005, lr_end=0.000001,
            time_limit=time_limit - (time.time() - t_start) - 2
        )
    
    if best_centers is None:
        best_centers = _init_grid()
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
