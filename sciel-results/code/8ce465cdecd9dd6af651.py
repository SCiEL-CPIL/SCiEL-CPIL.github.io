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

def _energy_and_grad(x, target_r, temperature=0.01):
    """
    Energy function: minimize violations of constraints that all pairs are at least 2r apart
    and all centers are at least r from walls.
    Returns (energy, gradient).
    """
    centers = x.reshape(N, 3)
    T2 = temperature**2
    
    # Pairwise energy: soft penalty for pairs closer than 2r
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    min_pair_dist_sq = (2.0 * target_r)**2
    
    # Soft-min of (sq_dist - min_pair_dist_sq) for pairs
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        # We want to penalize when sq_dist < min_pair_dist_sq
        # Use soft-min of (min_pair_dist_sq - sq_dist) which is positive when violated
        violations = min_pair_dist_sq - sq_dists
        max_viol = np.max(violations[finite_mask])
        exp_terms = np.zeros_like(violations)
        exp_terms[finite_mask] = np.exp((violations[finite_mask] - max_viol) / T2)
        sum_exp = np.sum(exp_terms)
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_max_violation = max_viol + T2 * log_sum
        else:
            soft_max_violation = max_viol
    else:
        soft_max_violation = 0.0
    
    # Wall energy: soft penalty for centers too close to walls
    wall_dists = 0.5 - np.abs(centers)
    violations_wall = target_r - wall_dists
    max_viol_wall = np.max(violations_wall)
    exp_wall = np.exp((violations_wall - max_viol_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        log_sum_wall = np.log(sum_wall)
        soft_max_viol_wall = max_viol_wall + T2 * log_sum_wall
    else:
        soft_max_viol_wall = max_viol_wall
    
    # Combined energy
    energy = soft_max_violation + soft_max_viol_wall
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        # d(soft_max)/dx_i = sum_j weights_ij * d(sq_dist_ij)/dx_i
        # d(sq_dist_ij)/dx_i = 2*(x_i - x_j)
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        grad_pair = 4.0 * (centers * sum_w[:, np.newaxis] - weighted_sum_x)
        grad += grad_pair
    
    # Wall gradient
    if sum_wall > 0:
        weights_wall = exp_wall / sum_wall
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        # d(violations_wall)/dx = -sign(x)
        grad_wall = -weights_wall * signs
        grad += grad_wall
    
    return energy, -grad

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

def _optimize_energy(init_centers, target_r, temperature=0.01, max_iter=3000, time_limit=10.0):
    """Optimize to minimize energy for a target radius."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _energy_and_grad(x, target_r, temperature)
        return val, grad
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
    )
    
    return result.x.reshape(N, 3)

def _binary_search_radius(init_centers, r_low, r_high, time_limit=30.0):
    """Binary search for the maximum feasible radius."""
    start_time = time.time()
    best_r = r_low
    best_centers = init_centers.copy()
    
    for _ in range(15):  # ~15 iterations for good precision
        if time.time() - start_time > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a configuration with radius r_mid
        try:
            optimized = _optimize_energy(init_centers, r_mid, temperature=0.005, 
                                         max_iter=2000, time_limit=8.0)
            actual_r = _compute_radius(optimized)
            
            if actual_r >= r_mid - 1e-6:
                # Feasible, try larger radius
                r_low = r_mid
                best_r = actual_r
                best_centers = optimized.copy()
            else:
                # Not feasible, try smaller radius
                r_high = r_mid
        except Exception:
            r_high = r_mid
    
    return best_r, best_centers

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
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Coarse optimization with gradient descent from multiple starts
    inits = [grid]
    for seed in range(4):
        inits.append(_random_init(N, seed=seed))
    
    temps_coarse = [0.05, 0.03, 0.02]
    
    for temp in temps_coarse:
        if time.time() > deadline - 50.0:
            break
        
        for init in inits:
            if time.time() > deadline - 40.0:
                break
            
            try:
                optimized = _optimize_from_init(init, temperature=temp,
                                                max_iter=2000, time_limit=8.0)
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update inits with best found so far
        inits = [best_centers] + inits[:2]
    
    # Stage 2: Simulated annealing to escape local minima
    if time.time() < deadline - 40.0:
        try:
            sa_centers, sa_radius = _simulated_annealing(
                best_centers, 
                initial_temp=0.03, 
                final_temp=0.001,
                n_iterations=3000,
                time_limit=20.0,
                seed=123
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers.copy()
        except Exception:
            pass
    
    # Stage 3: Binary search on radius with energy minimization
    if time.time() < deadline - 30.0:
        # Start with a reasonable lower bound
        r_low = best_radius * 0.95
        r_high = best_radius * 1.15
        
        try:
            bs_r, bs_centers = _binary_search_radius(
                best_centers, r_low, r_high, time_limit=25.0
            )
            if bs_r > best_radius:
                best_radius = bs_r
                best_centers = bs_centers.copy()
        except Exception:
            pass
    
    # Stage 4: Fine polishing with gradient optimization
    if time.time() < deadline - 15.0:
        for temp in [0.005, 0.002, 0.001, 0.0005]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                optimized = _optimize_from_init(best_centers, temperature=temp,
                                                max_iter=3000, time_limit=6.0)
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            polished, polished_r = _local_polish(
                best_centers, 
                max_iter=5000, 
                time_limit=4.0
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
