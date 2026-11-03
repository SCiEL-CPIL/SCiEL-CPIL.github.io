import numpy as np
import time
import math

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    return score

def dynamics_simulate(points, time_budget):
    """
    Rigid-body dynamics simulation using Lennard-Jones-like repulsive forces.
    """
    start_time = time.time()
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Initial velocity zero
    velocities = np.zeros_like(points)
    
    # Parameters
    sigma = 0.1  # Interaction range
    epsilon = 1.0
    dt = 0.001
    damping = 0.99
    
    # Adaptive time step
    while time.time() - start_time < time_budget:
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        mask_nonzero = pair_dists > 1e-9
        if np.any(mask_nonzero):
            d = pair_dists[mask_nonzero]
            # LJ potential force: F = 24*epsilon*(2*(sigma/d)^12 - (sigma/d)^6) / d
            # Simplified repulsive only: F = 12*epsilon*(sigma/d)^6 / d
            # For packing, we want to maximize min distance, so we use a repulsive force that pushes apart when close.
            # Let's use a simpler force: F = k / d^3 * direction
            # Or just gradient of -1/d
            # Let's use F = (sigma/d)^6 * (1/d) * direction
            inv_d = 1.0 / d
            force_mag = (sigma * inv_d)**6 * inv_d
            v_ij = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] * inv_d[:, np.newaxis]
            
            f_vec = v_ij * force_mag[:, np.newaxis]
            
            np.add.at(forces, i_idx[mask_nonzero], f_vec)
            np.add.at(forces, j_idx[mask_nonzero], -f_vec)
        
        # Boundary force
        rad = np.linalg.norm(points, axis=1)
        # Force pushing inward when close to boundary
        # F = k * (1 - r) if r > threshold, else 0
        # Let's use a soft wall: F = k * (1 - r)^2 * inward_direction for r > 0.9
        threshold = 0.9
        mask_wall = rad > threshold
        if np.any(mask_wall):
            wall_dist = 1.0 - rad[mask_wall]
            # Force magnitude increases as we get closer to wall
            f_mag = 10.0 * (threshold - rad[mask_wall])**2
            v_r = points[mask_wall] / rad[mask_wall, np.newaxis]
            forces[mask_wall] -= v_r * f_mag[:, np.newaxis]
        
        # Update velocities
        velocities = velocities * damping + forces * dt
        
        # Limit velocity
        v_norms = np.linalg.norm(velocities, axis=1)
        max_v = 0.01
        mask_high_v = v_norms > max_v
        if np.any(mask_high_v):
            velocities[mask_high_v] *= max_v / v_norms[mask_high_v, np.newaxis]
        
        # Update positions
        points += velocities * dt
        
        # Project inside container
        rad_new = np.linalg.norm(points, axis=1)
        mask_out = rad_new > 1.0
        if np.any(mask_out):
            points[mask_out] /= rad_new[mask_out, np.newaxis]
            velocities[mask_out] = 0.0  # Stop moving if hit wall
        
        # Compute score
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Decay dt slightly to settle
        dt *= 0.999
        
    return best_points, best_score

def polish(points, time_budget):
    """
    Gradient-based polish to refine the packing.
    """
    start_time = time.time()
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Adaptive k and lr
    k = 100.0
    lr = 0.005
    
    # We will run until time_budget is exhausted
    while time.time() - start_time < time_budget:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        current_score = np.min(all_dists)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Soft min gradient
        shifted_dists = all_dists - current_score
        # Only consider constraints close to the minimum to focus effort
        mask_close = shifted_dists < 0.05
        if not np.any(mask_close):
            break
            
        close_dists = shifted_dists[mask_close]
        exp_weights = np.exp(-k * close_dists)
        sum_weights = np.sum(exp_weights)
        
        if sum_weights == 0:
             break
            
        weights_full = np.zeros_like(all_dists)
        weights_full[mask_close] = exp_weights / sum_weights
        
        grads = np.zeros_like(points)
        
        pair_weights = weights_full[:num_pairs]
        d_pair = dists_matrix[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            v_ij = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]
            w_ij = pair_weights[mask_nonzero, np.newaxis]
            forces = np.zeros((N, 2))
            np.add.at(forces, i_idx[mask_nonzero], v_ij * w_ij)
            np.add.at(forces, j_idx[mask_nonzero], -v_ij * w_ij)
            grads += forces
            
        bound_weights = weights_full[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-9:
            grads /= grad_norm
            points += lr * grads
            
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Decay lr and increase k slightly
        lr *= 0.995
        k *= 1.01
        
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization
    # Generate a hexagonal lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.9
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.9, 0.9, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Dynamics Simulation Phase
    # Use dynamics to settle the configuration
    dyn_time = time_limit * 0.5
    dyn_points, dyn_score = dynamics_simulate(best_points, dyn_time)
    
    if dyn_score > best_score:
        best_score = dyn_score
        best_points = dyn_points.copy()
    
    # 3. Polish Phase
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        final_points, final_score = polish(best_points, remaining_time)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
            
    return best_points
