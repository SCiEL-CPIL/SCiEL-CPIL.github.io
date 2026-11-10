import numpy as np
import time

N = 100

def _compute_radius(c):
    """
    Compute the largest common radius for given centers.
    c: (N, 3) array of centers.
    Returns: min(min_pair_dist/2, min_wall_dist)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _lj_energy_and_forces(c, sigma=0.22, epsilon=1.0, wall_sigma=0.5, wall_epsilon=5.0):
    """
    Compute Lennard-Jones energy and forces for sphere centers.
    Uses 12-6 LJ potential: V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6)
    Plus wall repulsion to keep spheres inside the cube.
    
    Returns: (energy, forces)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    
    # Pairwise LJ interactions
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-8, dists, 1e-8)
    
    # LJ potential: V = 4*eps*((sigma/r)^12 - (sigma/r)^6)
    sr6 = (sigma / dists_safe) ** 6
    sr12 = sr6 * sr6
    v_pair = 4.0 * epsilon * (sr12 - sr6)
    np.fill_diagonal(v_pair, 0.0)
    
    energy_pair = np.sum(v_pair)
    
    # LJ force: F = -dV/dr * r_hat = 24*eps*(2*sigma^12/r^13 - sigma^6/r^7) * r_hat
    # In vector form: F_ij = 24*eps*(2*sr12 - sr6) * (sigma^6/r^7) * diff_ij
    # Simplified: F_ij = 24*eps*(2*sr12 - sr6) * (sigma^6) * diff_ij / r^7
    # Note: diff_ij = c_i - c_j, so force on i is in direction of diff_ij (repulsive when close)
    
    # Force magnitude factor: 24*eps*(2*sr12 - sr6) * sigma^6 / r^7
    # = 24*eps*(2*sr12 - sr6) * sigma^6 / (dists_sq * dists_safe)
    force_factor = 24.0 * epsilon * (2.0 * sr12 - sr6) * (sigma ** 6) / (dists_sq * dists_safe)
    np.fill_diagonal(force_factor, 0.0)
    
    # Force on particle i: sum_j force_factor[i,j] * diff[i,j]
    forces_pair = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    
    # Wall repulsion: push away from walls
    # Use a soft wall potential: V_wall = wall_epsilon * exp(-wall_sigma / d_wall) where d_wall is distance to wall
    # Or simpler: quadratic penalty when too close to wall
    wall_dist = 0.5 - np.abs(c)
    # Use exponential repulsion for smoothness
    wall_repulsion_strength = 10.0
    forces_wall = np.zeros_like(c)
    for i in range(n):
        for k in range(3):
            d = wall_dist[i, k]
            if d < 0.15:
                # Strong repulsion when close to wall
                force_mag = wall_repulsion_strength * (0.15 - d) ** 2 / (d + 1e-8)
                # Direction: towards center (opposite to sign of c[i,k])
                if c[i, k] > 0:
                    forces_wall[i, k] -= force_mag
                elif c[i, k] < 0:
                    forces_wall[i, k] += force_mag
    
    energy_wall = np.sum(wall_repulsion_strength * (0.15 - np.maximum(wall_dist, 0)) ** 3 / 3.0)
    
    total_energy = energy_pair + energy_wall
    total_forces = forces_pair + forces_wall
    
    return total_energy, total_forces

def _smooth_radius_surrogate(c, beta=50.0):
    """
    Smooth surrogate for the radius using log-sum-exp soft-min.
    Maximizes a smooth approximation of min(min_pair_dist/2, min_wall_dist).
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    wall_dists_flat = wall_dists.flatten()
    
    # Combine all constraints into a single array
    # We want to maximize t such that:
    #   t <= dists[i,j]/2 for all i<j
    #   t <= wall_dists[i,k] for all i,k
    # Use soft-min: -1/beta * log(sum(exp(-beta * constraints)))
    
    # Pair constraints: dists[i,j]/2 for i<j
    iu = np.triu_indices(n, k=1)
    pair_constraints = dists[iu] / 2.0
    
    # Wall constraints
    wall_constraints = wall_dists_flat
    
    # Combine
    all_constraints = np.concatenate([pair_constraints, wall_constraints])
    
    # Soft-min
    min_constraint = np.min(all_constraints)
    # Shift for numerical stability
    shifted = -beta * (all_constraints - min_constraint)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = min_constraint - log_sum_exp / beta
    
    return soft_min

def _objective_and_grad(c, beta=50.0):
    """
    Compute objective (negative of smooth radius) and gradient.
    We minimize -smooth_radius, which equals maximizing smooth_radius.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0, np.zeros_like(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    wall_dists_flat = wall_dists.flatten()
    
    # Constraints
    iu = np.triu_indices(n, k=1)
    pair_constraints = dists[iu] / 2.0
    wall_constraints = wall_dists_flat
    all_constraints = np.concatenate([pair_constraints, wall_constraints])
    
    # Soft-min computation
    min_c = np.min(all_constraints)
    shifted = -beta * (all_constraints - min_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = np.log(sum_exp)
    soft_min = min_c - log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. each constraint
    # d(soft_min)/d(constraint_k) = -exp_shifted[k] / sum_exp
    weights = -exp_shifted / sum_exp
    
    # Gradient w.r.t. c
    grad = np.zeros_like(c)
    
    # Pair constraints gradient
    # d(pair_constraints[m])/d(c) where pair_constraints[m] = dists[i,j]/2
    # d(dists[i,j])/d(c[i]) = (c[i] - c[j]) / dists[i,j]
    # d(dists[i,j])/d(c[j]) = (c[j] - c[i]) / dists[i,j]
    
    pair_weights = weights[:len(pair_constraints)]
    for m in range(len(iu[0])):
        i, j = iu[0][m], iu[1][m]
        w = pair_weights[m]
        d = dists[i, j]
        if d > 1e-8:
            direction = (c[i] - c[j]) / d
            grad[i] += w * 0.5 * direction
            grad[j] += w * 0.5 * (-direction)
    
    # Wall constraints gradient
    # wall_dists[i,k] = 0.5 - |c[i,k]|
    # d/d(c[i,k]) = -sign(c[i,k])
    wall_weights = weights[len(pair_constraints):]
    for idx in range(n * 3):
        i = idx // 3
        k = idx % 3
        w = wall_weights[idx]
        if c[i, k] > 0:
            grad[i, k] += w * (-1.0)
        elif c[i, k] < 0:
            grad[i, k] += w * (1.0)
    
    # We minimize -soft_min, so objective = -soft_min, grad = -grad
    objective = -soft_min
    grad = -grad
    
    return objective, grad

def _optimize(c_init, time_limit):
    """
    Lennard-Jones Potential with Simulated Annealing.
    c_init: Initial centers (N, 3)
    time_limit: Seconds to spend on this optimization attempt.
    Returns: Optimized centers, best radius found.
    """
    c = c_init.copy()
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    # LJ parameters
    # sigma should be related to desired spacing. For r~0.11, spacing~0.22
    sigma = 0.22
    epsilon = 1.0
    
    # Simulated annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    
    # Initial step size for gradient-based moves
    step_size = 0.005
    
    # Track acceptance for adaptive stepping
    accept_count = 0
    total_count = 0
    
    while time.time() - start_time < time_limit:
        
        # 1. Compute LJ energy and forces
        E, forces = _lj_energy_and_forces(c, sigma=sigma, epsilon=epsilon)
        
        # 2. Gradient-based move (deterministic)
        # Normalize forces to get stable steps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 5.0
        forces_normalized = np.where(force_norms > max_force, forces / force_norms * max_force, forces)
        
        # Gradient descent step
        c_grad = c - step_size * forces_normalized
        
        # 3. Add thermal noise (Monte Carlo move)
        noise = np.random.randn(n, 3) * np.sqrt(2 * T * step_size)
        c_new = c_grad + noise
        
        # 4. Clip to cube
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # 5. Compute new energy
        E_new, _ = _lj_energy_and_forces(c_new, sigma=sigma, epsilon=epsilon)
        
        # 6. Metropolis acceptance criterion
        dE = E_new - E
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            c = c_new
            accept_count += 1
            # Evaluate radius
            r_new = _compute_radius(c)
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            # Reject: stay at current position
            pass
        
        total_count += 1
        
        # 7. Adaptive step size and temperature
        # Adjust step size based on acceptance rate
        if total_count % 100 == 0:
            accept_rate = accept_count / total_count
            if accept_rate > 0.8:
                step_size *= 1.1  # Increase step
                accept_count = 0
                total_count = 0
            elif accept_rate < 0.2:
                step_size *= 0.9  # Decrease step
                accept_count = 0
                total_count = 0
        
        # Cool down
        T *= alpha
        
        # If temperature is too low, we're in deterministic mode
        if T < T_min:
            T = T_min
            # Reduce step size for fine-tuning
            step_size *= 0.999
            
            # If step size is very small, break
            if step_size < 1e-8:
                break
    
    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    Uses Lennard-Jones potential with simulated annealing.
    """
    total_time_limit = 115.0  # Safety margin from 150s limit
    start_time = time.time()
    
    best_c = None
    best_r = -1.0
    
    # Strategy 1: Grid initialization
    # 100 spheres. 4x5x5 grid? 4*5*5 = 100.
    nx, ny, nz = 4, 5, 5
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid_points = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    c_grid = grid_points[:N].copy()
    
    # Run optimization on grid start
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 10:
        c_opt, r_opt = _optimize(c_grid, t_left - 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Strategy 2: Random initialization
    c_rand = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 10:
        c_opt, r_opt = _optimize(c_rand, t_left - 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 3: Perturbed Grid
    c_perturbed = c_grid + np.random.randn(N, 3) * 0.03
    c_perturbed = np.clip(c_perturbed, -0.5, 0.5)
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 10:
        c_opt, r_opt = _optimize(c_perturbed, t_left - 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 4: Second Random
    c_rand2 = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 10:
        c_opt, r_opt = _optimize(c_rand2, t_left - 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 5: Final refinement on best solution
    if best_c is not None:
        t_left = total_time_limit - (time.time() - start_time)
        if t_left > 5:
            c_refine = best_c.copy()
            c_opt, r_opt = _optimize(c_refine, t_left - 1.0)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()

    if best_c is None:
        # Fallback: return grid
        return c_grid
        
    return best_c

# EVOLVE-BLOCK-END
