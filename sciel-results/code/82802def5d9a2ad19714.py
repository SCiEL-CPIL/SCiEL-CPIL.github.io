import numpy as np
import time
import scipy.optimize

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

def _objective_and_grad(c, beta=50.0):
    """
    Compute objective (negative of smooth radius) and gradient using vectorized ops.
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
    
    # Gradient w.r.t. c (vectorized)
    grad = np.zeros_like(c)
    
    # Pair constraints gradient - vectorized
    pair_weights = weights[:len(pair_constraints)]
    i_idx, j_idx = iu[0], iu[1]
    d_ij = dists[i_idx, j_idx]
    safe_d = np.where(d_ij > 1e-8, d_ij, 1.0)
    dir_vec = (c[i_idx] - c[j_idx]) / safe_d[:, np.newaxis]
    w_col = (pair_weights * 0.5)[:, np.newaxis]
    
    # grad[i] += w * 0.5 * direction
    np.add.at(grad, i_idx, w_col * dir_vec)
    # grad[j] += w * 0.5 * (-direction)
    np.add.at(grad, j_idx, -w_col * dir_vec)
    
    # Wall constraints gradient - vectorized
    wall_weights = weights[len(pair_constraints):]
    signs = np.sign(c)
    signs[signs == 0] = 1.0  # handle zero case
    wall_grad = -wall_weights.reshape(n, 3) * signs
    grad += wall_grad
    
    # We minimize -soft_min, so objective = -soft_min, grad = -grad
    objective = -soft_min
    grad = -grad
    
    return objective, grad

def _force_refine(c, time_limit):
    """
    Force-directed refinement: push overlapping pairs apart and pull toward center.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    start_time = time.time()
    
    lr = 0.002
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Repulsion between pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.25:  # Only interact if somewhat close
                    if d > 1e-8:
                        direction = (c[i] - c[j]) / d
                        strength = 1.0 / (d * d + 1e-6)
                        forces[i] += direction * strength
                        forces[j] -= direction * strength
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if c[i, k] > 0.4:
                    forces[i, k] -= 0.5
                elif c[i, k] < -0.4:
                    forces[i, k] += 0.5
        
        # Update positions
        c += lr * forces
        c = np.clip(c, -0.5, 0.5)
        
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Decay learning rate
        lr *= 0.995
    
    return best_c, best_r

def _optimize(c_init, time_limit, beta=50.0):
    """
    Gradient-based optimization using L-BFGS-B on the smooth surrogate.
    c_init: Initial centers (N, 3)
    time_limit: Seconds to spend on this optimization attempt.
    Returns: Optimized centers, best radius found.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    # Bounds for L-BFGS-B
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def callback(xk):
        nonlocal best_c, best_r
        r = _compute_radius(xk.reshape(N, 3))
        if r > best_r:
            best_r = r
            best_c = xk.reshape(N, 3).copy()
            
    try:
        # Use L-BFGS-B
        def obj_flat(x):
            c_2d = x.reshape(N, 3)
            val, grad = _objective_and_grad(c_2d, beta)
            return val, grad.flatten()
            
        result = scipy.optimize.minimize(
            obj_flat, 
            c.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            callback=callback,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < obj_flat(c.flatten())[0]:
            c_opt = result.x.reshape(N, 3)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
                
    except Exception as e:
        pass

    # Force-directed refinement
    t_left = time_limit - (time.time() - start_time)
    if t_left > 2.0:
        c_fd, r_fd = _force_refine(best_c, t_left - 1.0)
        if r_fd > best_r:
            best_r = r_fd
            best_c = c_fd.copy()

    # Final local refinement with exact radius
    t_left = time_limit - (time.time() - start_time)
    if t_left > 1.0:
        c = best_c.copy()
        step = 0.003
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(N):
                for d in range(3):
                    for sign in [-1, 1]:
                        c_new = c.copy()
                        c_new[i, d] += sign * step
                        if c_new[i, d] < -0.5 or c_new[i, d] > 0.5:
                            continue
                        r_new = _compute_radius(c_new)
                        if r_new > best_r:
                            c = c_new
                            best_c = c_new.copy()
                            best_r = r_new
                            improved = True
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break

    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    """
    total_time_limit = 115.0  # Safety margin from 120s
    start_time = time.time()
    
    best_c = None
    best_r = -1.0
    
    # Strategy 1: Grid initialization
    nx, ny, nz = 4, 5, 5
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid_points = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    c_grid = grid_points[:N].copy()
    
    # Run optimization on grid start
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_grid, t_left - 1.0, beta=50.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 2: Random initialization
    c_rand = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_rand, t_left - 1.0, beta=50.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 3: Perturbed Grid
    c_perturbed = c_grid + np.random.randn(N, 3) * 0.05
    c_perturbed = np.clip(c_perturbed, -0.5, 0.5)
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_perturbed, t_left - 1.0, beta=50.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 4: Different Beta
    if best_c is not None:
        t_left = total_time_limit - (time.time() - start_time)
        if t_left > 5:
             c_opt, r_opt = _optimize(best_c, t_left - 1.0, beta=100.0)
             if r_opt > best_r:
                 best_r = r_opt
                 best_c = c_opt.copy()

    # If we still have time, do a final polish on the best solution
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 2.0 and best_c is not None:
        c_opt, r_opt = _optimize(best_c, t_left - 0.5, beta=100.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    if best_c is None:
        return c_grid
        
    return best_c

# EVOLVE-BLOCK-END
