"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def _initial_jittered_grid(n):
    """Generate an initial jittered grid of n points in [-0.5, 0.5]^3."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    # Ensure we have at least n points
    while m**3 < n:
        m += 1
    
    # Create grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add jitter
    np.random.seed(42)
    jitter_scale = 0.5 / m * 0.8
    grid = grid + np.random.uniform(-jitter_scale, jitter_scale, size=grid.shape)
    
    # Clip to bounds
    grid = np.clip(grid, -0.5, 0.5)
    
    return grid[:n]

def _compute_radius(pts):
    """Compute the maximum feasible radius for the given points."""
    n = len(pts)
    # Boundary constraints
    max_abs = np.max(np.abs(pts))
    boundary_r = 0.5 - max_abs
    
    # Pairwise constraints
    # Calculate squared distances to avoid sqrt until necessary
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dists_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dists_sq)
    pairwise_r = min_dist / 2.0
    
    return min(boundary_r, pairwise_r)

def construct():
    n = N
    # Initial points
    pts = _initial_jittered_grid(n).copy()
    
    # Initial radius
    current_r = _compute_radius(pts)
    
    # Relaxation parameters
    # We will try to increase the radius iteratively
    target_r = current_r * 1.01
    max_iter = 5000
    step_size = 0.01
    
    # Soft sphere relaxation loop
    for step in range(max_iter):
        # Decay step size
        step_size = 0.01 * (1.0 - step / max_iter) ** 2 + 0.0001
        
        # Calculate forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # We want to push points apart if they are closer than 2 * target_r
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Threshold squared distance
        thresh_sq = (2.0 * target_r) ** 2
        
        # Mask for pairs closer than threshold
        mask = dists_sq < thresh_sq
        
        # Compute distances for masked pairs
        dists = np.sqrt(dists_sq)
        # Avoid division by zero
        dists[mask] = np.where(dists[mask] > 1e-6, dists[mask], 1e-6)
        
        # Force magnitude: proportional to overlap
        # F = k * (2*r - d) / d
        # We use a simple linear repulsion
        overlap = 2.0 * target_r - dists
        # Only apply force where overlap > 0
        overlap = np.where(mask, overlap, 0.0)
        
        # Unit vectors
        # diff / dists
        # Handle dists=0
        unit_vecs = diff / np.where(dists > 1e-6, dists, 1.0)[:, :, np.newaxis]
        
        # Force vector = unit_vecs * overlap
        force_vecs = unit_vecs * overlap[:, :, np.newaxis]
        
        # Sum forces for each point
        # forces[i] = sum_j force_vecs[i,j]
        forces += np.sum(force_vecs, axis=1)
        
        # Boundary repulsion
        # If a point is within target_r of a wall, push it in
        # x < -0.5 + target_r -> force_x > 0
        # x > 0.5 - target_r -> force_x < 0
        # Same for y, z
        
        # Calculate distance to each wall
        # Walls at -0.5 and 0.5
        # d_left = x - (-0.5) = x + 0.5
        # d_right = 0.5 - x
        
        # Force from left wall (if close)
        d_left = pts[:, 0] + 0.5
        mask_left = d_left < target_r
        if np.any(mask_left):
            forces[mask_left, 0] += (target_r - d_left[mask_left])
            
        # Force from right wall (if close)
        d_right = 0.5 - pts[:, 0]
        mask_right = d_right < target_r
        if np.any(mask_right):
            forces[mask_right, 0] -= (target_r - d_right[mask_right])
            
        # Y axis
        d_left_y = pts[:, 1] + 0.5
        mask_left_y = d_left_y < target_r
        if np.any(mask_left_y):
            forces[mask_left_y, 1] += (target_r - d_left_y[mask_left_y])
            
        d_right_y = 0.5 - pts[:, 1]
        mask_right_y = d_right_y < target_r
        if np.any(mask_right_y):
            forces[mask_right_y, 1] -= (target_r - d_right_y[mask_right_y])
            
        # Z axis
        d_left_z = pts[:, 2] + 0.5
        mask_left_z = d_left_z < target_r
        if np.any(mask_left_z):
            forces[mask_left_z, 2] += (target_r - d_left_z[mask_left_z])
            
        d_right_z = 0.5 - pts[:, 2]
        mask_right_z = d_right_z < target_r
        if np.any(mask_right_z):
            forces[mask_right_z, 2] -= (target_r - d_right_z[mask_right_z])
            
        # Update positions
        pts += step_size * forces
        
        # Clip to bounds to ensure validity
        pts = np.clip(pts, -0.5, 0.5)
        
        # Check current radius
        # We can check less frequently to save time
        if step % 100 == 0:
            current_r = _compute_radius(pts)
            # If we have exceeded our target, we can increase it
            # But we need to be careful not to overshoot too fast
            if current_r > target_r * 0.99:
                # Increase target slightly
                target_r = min(current_r * 1.005, 0.115) # Cap at known best + margin
            else:
                # If we are stuck, maybe decrease target to allow relaxation?
                # Or just continue with current target.
                pass

    # Final refinement: local search on the best points found
    # The relaxation might end up with a valid configuration.
    # We can do a few iterations of "push apart" with a fixed small radius.
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Try to improve with a few more relaxation steps with a slightly higher target
    # if we are close to the best known
    if best_r < 0.1113823475125:
        target_r = min(best_r * 1.01, 0.1113823475125)
        for step in range(2000):
            step_size = 0.001 * (1.0 - step / 2000) ** 2 + 0.00001
            
            forces = np.zeros_like(best_pts)
            
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            
            thresh_sq = (2.0 * target_r) ** 2
            mask = dists_sq < thresh_sq
            
            dists = np.sqrt(dists_sq)
            dists[mask] = np.where(dists[mask] > 1e-6, dists[mask], 1e-6)
            
            overlap = 2.0 * target_r - dists
            overlap = np.where(mask, overlap, 0.0)
            
            unit_vecs = diff / np.where(dists > 1e-6, dists, 1.0)[:, :, np.newaxis]
            force_vecs = unit_vecs * overlap[:, :, np.newaxis]
            forces += np.sum(force_vecs, axis=1)
            
            # Boundaries
            d_left = best_pts[:, 0] + 0.5
            mask_left = d_left < target_r
            if np.any(mask_left):
                forces[mask_left, 0] += (target_r - d_left[mask_left])
            d_right = 0.5 - best_pts[:, 0]
            mask_right = d_right < target_r
            if np.any(mask_right):
                forces[mask_right, 0] -= (target_r - d_right[mask_right])
                
            d_left_y = best_pts[:, 1] + 0.5
            mask_left_y = d_left_y < target_r
            if np.any(mask_left_y):
                forces[mask_left_y, 1] += (target_r - d_left_y[mask_left_y])
            d_right_y = 0.5 - best_pts[:, 1]
            mask_right_y = d_right_y < target_r
            if np.any(mask_right_y):
                forces[mask_right_y, 1] -= (target_r - d_right_y[mask_right_y])
                
            d_left_z = best_pts[:, 2] + 0.5
            mask_left_z = d_left_z < target_r
            if np.any(mask_left_z):
                forces[mask_left_z, 2] += (target_r - d_left_z[mask_left_z])
            d_right_z = 0.5 - best_pts[:, 2]
            mask_right_z = d_right_z < target_r
            if np.any(mask_right_z):
                forces[mask_right_z, 2] -= (target_r - d_right_z[mask_right_z])
                
            best_pts += step_size * forces
            best_pts = np.clip(best_pts, -0.5, 0.5)
            
            if step % 100 == 0:
                r_check = _compute_radius(best_pts)
                if r_check > best_r:
                    best_r = r_check
                    target_r = min(r_check * 1.005, 0.1113823475125)

    # Ensure points are strictly inside
    eps = 1e-10
    best_pts = np.clip(best_pts, -0.5 + eps, 0.5 - eps)
    
    return best_pts
# EVOLVE-BLOCK-END
