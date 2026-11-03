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

def _soft_min_radius(centers, temperature=0.001):
    """
    Smooth surrogate for the minimum radius.
    Uses log-sum-exp trick for soft-min of pairwise distances/2 and wall distances.
    
    r_soft = -1/temperature * log(sum(exp(-temperature * d_i / 2)) + exp(-temperature * w_i))
    where d_i are pairwise distances and w_i are wall distances.
    """
    if len(centers) < 2:
        wall_dists = 0.5 - np.abs(centers)
        # soft-min of wall distances
        min_w = np.min(wall_dists)
        log_sum = np.log(np.sum(np.exp(-(wall_dists - min_w) / temperature)))
        return -(min_w + temperature * log_sum)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, 0)
    
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0  # radius from pairwise
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # Soft-min: -1/T * log(sum(exp(-T * d_i)))
    min_d = np.min(all_dists)
    log_sum = np.log(np.sum(np.exp(-(all_dists - min_d) / temperature)))
    soft_min = -(min_d + temperature * log_sum)
    
    return float(soft_min)

def _objective_and_grad(centers_flat, temperature=0.001):
    """
    Objective: minimize -soft_min_radius (i.e., maximize soft-min radius).
    Returns (value, gradient).
    """
    centers = centers_flat.reshape(N, 3)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, 0)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    all_dists = np.concatenate([pair_dists, wall_dists.ravel()])
    
    min_d = np.min(all_dists)
    exp_terms = np.exp(-(all_dists - min_d) / temperature)
    log_sum = np.log(np.sum(exp_terms))
    soft_min = -(min_d + temperature * log_sum)
    
    # Gradient of soft-min w.r.t. centers
    # d(soft_min)/d(center_k) = -sum_i (exp_term_i / sum_exp) * d(d_i)/d(center_k)
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 3))
    
    # Pairwise contribution
    # For each pair (i,j), d_ij = |c_i - c_j| / 2
    # d(d_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(d_ij)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = dists[i, j]
        if d > 1e-10:
            direction = (centers[i] - centers[j]) / d
            grad[i] += w * direction / 2.0
            grad[j] -= w * direction / 2.0
    
    # Wall contribution
    # d(wall_dist_k)/d(center_k) = -sign(center_k) for each coordinate
    for i in range(N):
        for k in range(3):
            idx = len(pair_dists) + i * 3 + k
            w = weights[idx]
            if centers[i, k] > 0:
                grad[i, k] -= w
            elif centers[i, k] < 0:
                grad[i, k] += w
    
    # We're minimizing -soft_min, so gradient is -d(soft_min)/d(centers)
    # But we want to maximize soft_min, so we minimize negative soft_min
    value = -soft_min
    grad = -grad  # gradient of -soft_min
    
    return value, grad.ravel()

def _optimize_from_init(centers_init, target_temp_start=0.01, target_temp_end=0.0001, 
                        max_iter=500, time_budget=20.0):
    """
    Optimize from initial centers using L-BFGS-B with annealing temperature.
    """
    start_time = time.time()
    centers = centers_init.copy().flatten()
    
    # Annealing schedule: decrease temperature
    n_anneal_steps = 5
    temps = np.geomspace(target_temp_start, target_temp_end, n_anneal_steps)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers.reshape(N, 3))
    
    for temp in temps:
        if time.time() - start_time > time_budget:
            break
        
        result = minimize(
            lambda x: _objective_and_grad(x, temp)[0],
            centers,
            jac=lambda x: _objective_and_grad(x, temp)[1],
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (N * 3),
            options={'maxiter': max_iter // n_anneal_steps, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        centers = result.x
        r = _compute_radius(centers.reshape(N, 3))
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    return best_centers.reshape(N, 3), best_radius

def _polish_critical_pairs(centers, time_budget=10.0):
    """
    Final polish: focus on the most constrained pairs and walls.
    Use a higher temperature for smoothness, then refine.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for _ in range(3):
        if time.time() - start_time > time_budget:
            break
        
        # Use a moderate temperature for smoothing
        temp = 0.0005
        result = minimize(
            lambda x: _objective_and_grad(x, temp)[0],
            centers.flatten(),
            jac=lambda x: _objective_and_grad(x, temp)[1],
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        centers = result.x
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
    
    return centers, best_radius

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

def _jitter_init(n, base, amount=0.02, seed=None):
    """Add jitter to a base configuration."""
    rng = np.random.RandomState(seed)
    return base + rng.uniform(-amount, amount, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Strategy 1: Start with grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Try multiple initializations and optimize
    inits = [('grid', grid)]
    for seed in range(8):
        inits.append((f'rand_{seed}', _random_init(N, seed=seed)))
    
    # Optimize from each initialization
    for name, init in inits:
        if time.time() > deadline - 30.0:
            break
        
        try:
            optimized, r_opt = _optimize_from_init(
                init, 
                target_temp_start=0.01, 
                target_temp_end=0.0001,
                max_iter=300,
                time_budget=12.0
            )
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            continue
    
    # Final polish on the best configuration
    if time.time() < deadline - 15.0:
        try:
            polished, r_polished = _polish_critical_pairs(
                best_centers, 
                time_budget=12.0
            )
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
        except Exception:
            pass
    
    # Additional polish with smaller temperature
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, 0.00005)[0],
                best_centers.flatten(),
                jac=lambda x: _objective_and_grad(x, 0.00005)[1],
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            final_centers = result.x.reshape(N, 3)
            r_final = _compute_radius(final_centers)
            if r_final > best_radius:
                best_radius = r_final
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _grid_init(N)
    
    # Final validation: clip to valid range
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
