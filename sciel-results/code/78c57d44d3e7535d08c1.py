import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initialization: 5x5x4 grid with jitter
    # 100 points. 5*5*4 = 100.
    # Grid coords in [-0.5, 0.5]
    nx, ny, nz = 5, 5, 4
    # Ensure we have exactly N points
    # If N is not nx*ny*nz, we need to handle it, but 100 fits perfectly.
    
    # Create grid
    x_coords = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    y_coords = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    z_coords = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    # Meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    pos = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add jitter
    # Jitter magnitude should be small, e.g., 10% of grid spacing
    grid_spacing = 1.0 / max(nx, ny, nz)
    pos += np.random.normal(0, 0.1 * grid_spacing, pos.shape)
    
    # Ensure inside cube
    pos = np.clip(pos, -0.5, 0.5)
    
    best_pos = pos.copy()
    best_radius = 0.0
    
    # Optimization Parameters
    alpha = 1.0
    alpha_max = 100.0
    step_size = 0.01
    min_step_size = 1e-5
    decay_alpha = 1.001
    decay_step = 0.999
    
    # Precompute indices for pairs
    # We will use full matrix operations for speed
    
    while time.time() < deadline:
        
        # Compute pairwise distances
        # pos is (N, 3)
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        
        # Vectorized distance calculation
        sq_dists = np.sum(pos**2, axis=1)[:, np.newaxis] + np.sum(pos**2, axis=1)[np.newaxis, :] - 2 * pos @ pos.T
        np.maximum(sq_dists, 0, out=sq_dists) # numerical stability
        dists = np.sqrt(sq_dists)
        
        # Mask diagonal (self distance)
        np.fill_diagonal(dists, np.inf)
        
        # Compute Wall distances
        # w_ik = 0.5 - |pos_ik|
        abs_pos = np.abs(pos)
        wall_dists = 0.5 - abs_pos # (N, 3)
        
        # Compute Soft Min components
        # E = sum(e^{-alpha * dists}) + sum(e^{-alpha * wall_dists})
        
        # To avoid overflow/underflow issues with large alpha, 
        # we can normalize or just be careful. 
        # exp(-100 * 0.1) is small. exp(-100 * 1e-6) is large.
        # It's safer to work with log-sum-exp, but for gradient we need the raw sums.
        # If alpha is large, only the minimum terms contribute significantly.
        
        # Calculate weights
        # Clip distances to avoid div by zero in gradient
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        safe_walls = np.where(wall_dists < 1e-9, 1e-9, wall_dists)
        
        w_pairs = np.exp(-alpha * safe_dists)
        w_walls = np.exp(-alpha * safe_walls)
        
        # Sum for E
        E = np.sum(w_pairs) + np.sum(w_walls)
        
        # If E is 0 or inf, break or reset
        if not np.isfinite(E) or E == 0:
             break
             
        # Compute Gradient
        # Grad for pairs:
        # G_pair[i, k] = sum_j w_pairs[i, j] * (pos[i, k] - pos[j, k]) / dists[i, j]
        
        # Create matrix of unit vectors
        # diff[i, j, k] = pos[i, k] - pos[j, k]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        # Normalize
        # unit_vecs[i, j, k] = diff[i, j, k] / dists[i, j]
        # Handle dists=0 (diagonal) by setting to 0
        safe_dists_3d = np.where(dists < 1e-9, 1.0, safe_dists)
        unit_vecs = diff / safe_dists_3d[:, :, np.newaxis]
        
        # Zero out diagonal
        np.fill_diagonal(unit_vecs[:, :, 0], 0) # This doesn't work for 3D, need to zero the whole row/col in 3D
        # Better:
        mask_diag = np.eye(N, dtype=bool)
        unit_vecs[mask_diag] = 0.0
        
        # Weighted Sum
        # G_pairs[i, k] = sum_j w_pairs[i, j] * unit_vecs[i, j, k]
        G_pairs = np.einsum('ij,ijk->ik', w_pairs, unit_vecs)
        
        # Grad for walls:
        # G_walls[i, k] = w_walls[i, k] * sign(pos[i, k])
        signs = np.sign(pos)
        # If pos is 0, sign is 0, which is fine (derivative of |x| at 0 is undefined, but 0 is a good guess)
        G_walls = w_walls * signs
        
        # Total Gradient
        Grad = G_pairs + G_walls
        
        # Normalize Gradient?
        # The magnitude of Grad depends on alpha and E.
        # We want to take a step in the direction of Grad.
        # Step size control:
        # If we use fixed step, we might overshoot.
        # Let's normalize the gradient to unit length and scale by step_size?
        # Or just use step_size * Grad / (norm(Grad) + eps)?
        # Actually, the soft-min gradient magnitude is roughly 1 (since d/dx (-1/a ln(sum)) is bounded).
        # Let's try: pos += step_size * Grad
        
        # Update
        pos += step_size * Grad
        
        # Clip to cube
        pos = np.clip(pos, -0.5, 0.5)
        
        # Calculate current radius for tracking
        # r = min(min_pair_dist, min_wall_dist)
        min_pair = np.min(dists)
        min_wall = np.min(wall_dists)
        current_radius = min(min_pair, min_wall)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pos = pos.copy()
            
        # Annealing
        alpha *= decay_alpha
        if alpha > alpha_max:
            alpha = alpha_max
            
        step_size *= decay_step
        if step_size < min_step_size:
            step_size = min_step_size
            
        # Optional: Restart if stuck?
        # For now, just keep going.
        
    # Final Local Refinement
    # Try to push apart the closest pair
    # Identify closest pair
    dists_final = np.sqrt(np.maximum(np.sum(best_pos**2, axis=1)[:, np.newaxis] + np.sum(best_pos**2, axis=1)[np.newaxis, :] - 2 * best_pos @ best_pos.T, 0))
    np.fill_diagonal(dists_final, np.inf)
    i_min, j_min = np.unravel_index(np.argmin(dists_final), dists_final.shape)
    
    # Push them apart slightly if possible
    # Direction from j to i
    dir_vec = best_pos[i_min] - best_pos[j_min]
    norm = np.linalg.norm(dir_vec)
    if norm > 1e-9:
        dir_vec /= norm
        
        # Try small steps
        for delta in np.linspace(0.001, 0.01, 10):
            new_pos = best_pos.copy()
            new_pos[i_min] += delta * dir_vec
            new_pos[j_min] -= delta * dir_vec
            
            # Check validity
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            # Calculate new radius
            sq_d = np.sum(new_pos**2, axis=1)[:, np.newaxis] + np.sum(new_pos**2, axis=1)[np.newaxis, :] - 2 * new_pos @ new_pos.T
            np.maximum(sq_d, 0, out=sq_d)
            d = np.sqrt(sq_d)
            np.fill_diagonal(d, np.inf)
            
            abs_p = np.abs(new_pos)
            w = 0.5 - abs_p
            
            r_new = min(np.min(d), np.min(w))
            
            if r_new > best_radius:
                best_radius = r_new
                best_pos = new_pos
            else:
                break
                
    return best_pos
