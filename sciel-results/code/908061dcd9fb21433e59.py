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
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
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
    
    return float(soft_min_combined)

def _objective_and_grad(x, temperature=0.01):
    """Objective: maximize soft-min radius. Returns (-radius, gradient)."""
    centers = x.reshape(N, 3)
    
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
        weights = exp_terms / sum_exp
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        grad_pair_sq = 4.0 * (centers * sum_w[:, np.newaxis] - weighted_sum_x)
        
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        # Wall gradient
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        
        d_sum_wall_dx = exp_wall * signs / T2
        d_soft_min_wall_dx = -T2 * d_sum_wall_dx / sum_wall
        
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        
        grad = grad_combined
    
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

def _simulated_annealing(centers, initial_temp=0.05, final_temp=0.001, 
                          n_iterations=5000, time_limit=30.0, seed=42):
    """
    Simulated annealing to escape local minima.
    Uses the soft-min radius as energy (we want to maximize it).
    """
    rng = np.random.RandomState(seed)
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    # Temperature schedule
    alpha = (final_temp / initial_temp) ** (1.0 / n_iterations)
    temp = initial_temp
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random perturbation
        # Move one or a few spheres randomly
        n_moved = rng.randint(1, 4)
        indices = rng.choice(N, size=n_moved, replace=False)
        
        new_c = c.copy()
        for idx in indices:
            # Random perturbation with Gaussian noise
            step_size = temp * 0.5
            new_c[idx] += rng.normal(0, step_size, size=3)
        
        # Clip to valid range
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        # Evaluate
        r_new = _compute_radius(new_c)
        r_old = _compute_radius(c)
        
        # Accept if better, or with probability exp((r_new - r_old) / temp)
        if r_new > r_old:
            c = new_c
            if r_new > best_r:
                best_r = r_new
                best_c = new_c.copy()
        else:
            # Accept uphill move with probability
            delta = r_old - r_new
            if delta < 1e-10:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / (temp * 0.1))
            
            if rng.random() < accept_prob:
                c = new_c
        
        # Cool down
        temp *= alpha
    
    return best_c, best_r

def _local_polish(centers, target_r=None, max_iter=2000, time_limit=15.0):
    """
    Final high-precision polish using L-BFGS-B with very small temperature.
    """
    if target_r is None:
        target_r = _compute_radius(centers)
    
    # Use a very small temperature for precise optimization
    temps = [0.001, 0.0005, 0.0001]
    
    best_c = centers.copy()
    best_r = _compute_radius(best_c)
    
    for temp in temps:
        if time.time() > time_limit:
            break
        
        try:
            optimized = _optimize_from_init(best_c, temperature=temp,
                                            max_iter=max_iter, time_limit=time_limit)
            r_opt = _compute_radius(optimized)
            if r_opt > best_r:
                best_r = r_opt
                best_c = optimized.copy()
        except Exception:
            continue
    
    return best_c, best_r

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Strategy: Combine gradient-based optimization with simulated annealing
    # to escape local minima and find better packings
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Coarse optimization with gradient descent from multiple starts
    inits = [grid]
    for seed in range(3):
        inits.append(_random_init(N, seed=seed))
    
    temps_coarse = [0.05, 0.03, 0.02]
    
    for temp in temps_coarse:
        if time.time() > deadline - 40.0:
            break
        
        for init in inits:
            if time.time() > deadline - 30.0:
                break
            
            try:
                optimized = _optimize_from_init(init, temperature=temp,
                                                max_iter=2000, time_limit=10.0)
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update inits with best found so far
        inits = [best_centers] + inits[:2]
    
    # Stage 2: Simulated annealing to escape local minima
    if time.time() < deadline - 30.0:
        # Run simulated annealing from the best configuration found so far
        try:
            sa_centers, sa_radius = _simulated_annealing(
                best_centers, 
                initial_temp=0.03, 
                final_temp=0.001,
                n_iterations=3000,
                time_limit=25.0,
                seed=123
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers.copy()
        except Exception:
            pass
    
    # Stage 3: Fine polishing with gradient optimization
    if time.time() < deadline - 10.0:
        # Multiple polishing passes with decreasing temperature
        for temp in [0.005, 0.002, 0.001, 0.0005]:
            if time.time() > deadline - 5.0:
                break
            
            try:
                optimized = _optimize_from_init(best_centers, temperature=temp,
                                                max_iter=3000, time_limit=8.0)
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 4: Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            polished, polished_r = _local_polish(
                best_centers, 
                max_iter=5000, 
                time_limit=5.0
            )
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
