def _force_relaxation(centers, rng, time_budget):
    """
    Force-directed relaxation to maximize the minimum radius.
    Applies repulsive forces to close pairs and attractive forces to walls.
    Uses momentum for stability.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    velocity = np.zeros_like(current)
    momentum = 0.8
    learning_rate = 0.001
    # Target radius: we want to push towards a larger radius
    # Start with current radius + small epsilon
    target_r = best_r + 0.001
    
    # Adaptive learning rate
    while True:
        if time.time() - start_time > time_budget:
            break
            
        # Compute forces
        forces = np.zeros_like(current)
        
        # 1. Pairwise repulsion
        # We only care about pairs that are too close (d < 2 * target_r)
        diff = current[:, None, :] - current[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dists = np.sqrt(dist_sq)
        
        # Violation: d < 2 * target_r
        # Force magnitude proportional to violation
        # We want to push them apart
        # F_ij = (2 * target_r - d_ij) / d_ij * (p_i - p_j)
        mask = dists < 2.0 * target_r
        # Only compute for non-zero distances
        valid_mask = mask & (dists > 1e-6)
        
        if np.any(valid_mask):
            i_idx, j_idx = np.nonzero(valid_mask)
            d_vals = dists[i_idx, j_idx]
            p_i = current[i_idx]
            p_j = current[j_idx]
            
            # Force vector from j to i
            # Magnitude is (2*r - d)
            # Direction is (p_i - p_j) / d
            mag = (2.0 * target_r - d_vals)
            # Normalize direction
            dir_vec = (p_i - p_j) / d_vals[:, None]
            f_vec = mag[:, None] * dir_vec
            
            # Accumulate forces
            # f_vec is force on i from j
            # So force on i += f_vec
            # Force on j -= f_vec
            np.add.at(forces, i_idx, f_vec)
            np.add.at(forces, j_idx, -f_vec)
            
        # 2. Wall attraction (if too close to wall)
        # We want to keep circles inside [-0.5, 0.5]
        # If x < -0.5 + target_r, push right
        # If x > 0.5 - target_r, push left
        
        min_x = -0.5 + target_r
        max_x = 0.5 - target_r
        min_y = -0.5 + target_r
        max_y = 0.5 - target_r
        
        f_x_wall = np.zeros(N)
        f_y_wall = np.zeros(N)
        
        x = current[:, 0]
        y = current[:, 1]
        
        # Left wall
        mask_left = x < min_x
        f_x_wall[mask_left] += (min_x - x[mask_left])
        
        # Right wall
        mask_right = x > max_x
        f_x_wall[mask_right] -= (x[mask_right] - max_x)
        
        # Bottom wall
        mask_bottom = y < min_y
        f_y_wall[mask_bottom] += (min_y - y[mask_bottom])
        
        # Top wall
        mask_top = y > max_y
        f_y_wall[mask_top] -= (y[mask_top] - max_y)
        
        forces[:, 0] += f_x_wall
        forces[:, 1] += f_y_wall
        
        # Update velocity and position
        velocity = momentum * velocity + learning_rate * forces
        current += velocity
        
        # Clamp to square (hard constraint)
        current = np.clip(current, -0.5, 0.5)
        
        # Check radius
        curr_r = _compute_radius(current)
        
        if curr_r > best_r:
            best_r = curr_r
            best = current.copy()
            # If we improved, increase target radius slightly
            target_r = best_r + 0.001
        else:
            # If we didn't improve, maybe reduce learning rate or target?
            # Simple strategy: if stuck, reduce target to current best to stabilize
            # We check every 100 steps to avoid frequent branching
            # But we are in a while loop, so we need a counter
            # Let's just reduce learning rate slightly every time we fail?
            # No, that might get stuck.
            # Let's keep target_r = best_r + 0.001 always.
            # If we are stuck, it means we are in a local minimum of the force landscape.
            # Adding noise might help.
            if rng.random() < 0.01:
                current += rng.normal(0, 0.001, current.shape)
                current = np.clip(current, -0.5, 0.5)
        
    return best, best_r
