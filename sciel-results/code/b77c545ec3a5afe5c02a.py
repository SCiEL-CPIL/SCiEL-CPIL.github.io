<<<<<<< SEARCH
def _grow_and_push(positions, radii, max_iters=5000, grow_rate=0.001, push_iters=50):
    """Iteratively push circles apart and grow radii."""
    n = len(radii)
    positions = positions.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        # Check for violations and push
        violated = False
        for _ in range(push_iters):
            max_push = 0
            for i in range(n):
                # Boundary constraints
                if positions[i][0] < radii[i]:
                    positions[i][0] = radii[i]
                    violated = True
                    max_push = max(max_push, radii[i] - positions[i][0])
                if positions[i][0] > 1 - radii[i]:
                    positions[i][0] = 1 - radii[i]
                    violated = True
                    max_push = max(max_push, positions[i][0] - (1 - radii[i]))
                if positions[i][1] < radii[i]:
                    positions[i][1] = radii[i]
                    violated = True
                    max_push = max(max_push, radii[i] - positions[i][1])
                if positions[i][1] > 1 - radii[i]:
                    positions[i][1] = 1 - radii[i]
                    violated = True
                    max_push = max(max_push, positions[i][1] - (1 - radii[i]))
            
            # Pairwise overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist - 1e-12:
                        if dist < 1e-12:
                            # Random direction
                            angle = np.random.uniform(0, 2*np.pi)
                            dx, dy = np.cos(angle), np.sin(angle)
                            dist = 1.0
                        overlap = (min_dist - dist) / 2
                        nx, ny = dx/dist, dy/dist
                        positions[i][0] += overlap * nx
                        positions[i][1] += overlap * ny
                        positions[j][0] -= overlap * nx
                        positions[j][1] -= overlap * ny
                        violated = True
                        max_push = max(max_push, overlap)
            
            if not violated:
                break
        
        if violated:
            continue
        
        # Grow radii
        radii[:] *= (1 + grow_rate)
        
        # Check if still valid after growth
        valid = True
        for i in range(n):
            if positions[i][0] < radii[i] or positions[i][0] > 1 - radii[i]:
                valid = False
                break
            if positions[i][1] < radii[i] or positions[i][1] > 1 - radii[i]:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
        if not valid:
            # Shrink back
            radii[:] /= (1 + grow_rate)
            break
    
    return positions, radii
=======
def _grow_and_push(positions, radii, max_iters=2000, grow_rate=0.002, push_iters=10):
    """Iteratively push circles apart and grow radii using vectorized operations."""
    n = len(radii)
    positions = positions.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        # Check for violations and push
        violated = False
        for _ in range(push_iters):
            # Boundary constraints (vectorized)
            pos_x = positions[:, 0]
            pos_y = positions[:, 1]
            
            # Lower bounds
            mask_x_low = pos_x < radii
            mask_y_low = pos_y < radii
            if np.any(mask_x_low) or np.any(mask_y_low):
                violated = True
                positions[mask_x_low, 0] = radii[mask_x_low]
                positions[mask_y_low, 1] = radii[mask_y_low]
            
            # Upper bounds
            mask_x_high = pos_x > 1 - radii
            mask_y_high = pos_y > 1 - radii
            if np.any(mask_x_high) or np.any(mask_y_high):
                violated = True
                positions[mask_x_high, 0] = 1 - radii[mask_x_high]
                positions[mask_y_high, 1] = 1 - radii[mask_y_high]
            
            # Pairwise overlaps (vectorized)
            # Compute distance matrix
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq)
            min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Find overlaps (upper triangle only)
            i_idx, j_idx = np.triu_indices(n, k=1)
            overlaps = min_dist[i_idx, j_idx] - dist[i_idx, j_idx]
            
            if np.any(overlaps > 1e-12):
                violated = True
                # Only process overlapping pairs
                overlap_mask = overlaps > 1e-12
                i_ov = i_idx[overlap_mask]
                j_ov = j_idx[overlap_mask]
                ov_amt = overlaps[overlap_mask]
                
                # Get vectors
                d_vec = positions[i_ov] - positions[j_ov]
                d_norm = dist[i_ov, j_ov]
                
                # Avoid division by zero
                d_norm[d_norm < 1e-12] = 1e-12
                n_vec = d_vec / d_norm[:, np.newaxis]
                
                # Update positions
                # i moves by +0.5 * overlap * n_vec
                # j moves by -0.5 * overlap * n_vec
                push_i = 0.5 * ov_amt[:, np.newaxis] * n_vec
                push_j = -0.5 * ov_amt[:, np.newaxis] * n_vec
                
                # Accumulate updates (simple sum, might need more sophisticated relaxation)
                # For simplicity, apply directly. This is a basic projection.
                positions[i_ov] += push_i
                positions[j_ov] += push_j
            
            if not violated:
                break
        
        if violated:
            continue
        
        # Grow radii
        radii[:] *= (1 + grow_rate)
        
        # Check if still valid after growth (vectorized check)
        pos_x = positions[:, 0]
        pos_y = positions[:, 1]
        
        boundary_violation = np.any(pos_x < radii) or np.any(pos_x > 1 - radii) or \
                            np.any(pos_y < radii) or np.any(pos_y > 1 - radii)
        
        if boundary_violation:
            radii[:] /= (1 + grow_rate)
            break
            
        # Pairwise check
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Check upper triangle
        i_idx, j_idx = np.triu_indices(n, k=1)
        if np.any(min_dist[i_idx, j_idx] - dist[i_idx, j_idx] > 1e-12):
            radii[:] /= (1 + grow_rate)
            break
    
    return positions, radii
>>>>>>> REPLACE

