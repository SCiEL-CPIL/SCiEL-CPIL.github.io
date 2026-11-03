import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Initialize with a 4x4x4 grid subset
    # 4x4x4 = 64 points. We need 50.
    # The grid points are at (-0.375, -0.125, 0.125, 0.375)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We need to pick 50 points from the 64 grid points.
    # Taking the first 50 is fine, but we should ensure they are well-distributed.
    # Actually, removing 14 points from a 4x4x4 grid creates holes.
    # A better initial guess might be a dense packing.
    # Let's stick to the grid but shuffle to break symmetry in the optimization.
    
    best_radius = 0.0
    best_points = None
    
    # Run multiple restarts to escape local minima
    # We have ~110s. Let's allow a few restarts if they are fast.
    # However, a single long run with adaptive step size is often better for this problem.
    # Let's do one main run with adaptive step size, and if time permits, a second with different jitter.
    
    for restart in range(3):
        if time.time() > deadline:
            break
            
        # Initialize points
        if restart == 0:
            # Standard grid subset
            idx = np.arange(N)
            points = grid[idx].copy()
        else:
            # Jittered grid subset
            idx = np.random.choice(len(grid), N, replace=False)
            points = grid[idx].copy()
            points += np.random.uniform(-0.02, 0.02, points.shape)
            points = np.clip(points, -0.49, 0.49)
            
        # Adaptive Local Search Parameters
        step_size = 0.05
        patience = 0
        max_patience = 100 # Increase step size if no improvement for this many steps
        current_radius = _compute_radius(points)
        local_best_radius = current_radius
        local_best_points = points.copy()
        
        # Main Optimization Loop
        # We run until deadline or until we hit a very high patience (stuck)
        while time.time() < deadline:
            # 1. Compute Forces
            # Pairwise differences
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Wall distances
            # Distance to nearest wall for each point and dimension
            # points are in [-0.5, 0.5]
            # dist to -0.5: points + 0.5
            # dist to 0.5: 0.5 - points
            wall_dists = np.minimum(points + 0.5, 0.5 - points) # shape (N, 3)
            min_wall_dist = np.min(wall_dists, axis=1) # shape (N,)
            
            # Current Radius
            min_pair_dist = np.min(dist)
            r = min(min_pair_dist / 2.0, np.min(min_wall_dist))
            
            # Update Best
            if r > local_best_radius:
                local_best_radius = r
                local_best_points = points.copy()
                patience = 0
                # Decay step size slightly on improvement to refine
                step_size *= 0.98
                step_size = max(step_size, 0.001)
                
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
            
            # 2. Calculate Forces
            # Sphere-Sphere Repulsion
            # F_ij = (c_i - c_j) / dist_ij^2
            # Only apply force if dist is small enough to avoid long-range noise?
            # Actually, for packing, long-range repulsion helps fill space.
            # But 1/d^2 is strong.
            inv_dist_sq = 1.0 / (dist * dist + 1e-12)
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            force_spheres = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Wall Repulsion
            # If a point is close to a wall, push it away.
            force_walls = np.zeros_like(points)
            for k in range(3):
                d_left = points[:, k] + 0.5
                d_right = 0.5 - points[:, k]
                
                # Determine closest wall
                # If d_left < d_right, closest is left, push right (+1)
                # If d_right < d_left, closest is right, push left (-1)
                min_d = np.minimum(d_left, d_right)
                direction = np.where(d_left < d_right, 1.0, -1.0)
                
                # Force magnitude: 1 / min_d^2
                # Cap it to prevent instability
                mag = 1.0 / (min_d * min_d + 1e-12)
                mag = np.minimum(mag, 50.0) # Cap force
                
                force_walls[:, k] = direction * mag
            
            total_force = force_spheres + force_walls
            
            # 3. Update Positions
            # Normalize forces to prevent huge jumps
            force_mags = np.linalg.norm(total_force, axis=1)
            max_force = np.max(force_mags)
            if max_force > 1e-9:
                total_force = total_force / max_force
            
            points += step_size * total_force
            
            # Clip to cube (strictly inside to avoid boundary issues in next step)
            points = np.clip(points, -0.4999, 0.4999)
            
            # 4. Patience Logic
            if r <= local_best_radius:
                patience += 1
                if patience > max_patience:
                    # Stuck, increase step size to escape
                    step_size = min(0.1, step_size * 1.2)
                    patience = 0
                    # Add small random kick to break symmetry
                    points += np.random.randn(N, 3) * 0.01
                    points = np.clip(points, -0.4999, 0.4999)
            
            # Check if we are converging very slowly and time is running out
            if time.time() > deadline - 5.0:
                break
        
        # End of restart loop
        if time.time() > deadline:
            break
            
    # Final Safety Check
    if best_points is None:
        return grid[:N]
        
    # Ensure the returned points are valid
    # Re-clip to ensure no floating point errors pushed them out
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points

def _compute_radius(points):
    """
    Computes the maximum radius for a given set of points.
    Returns -1 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return -1.0
    
    # Check bounds
    if np.any(points < -0.5 - 1e-12) or np.any(points > 0.5 + 1e-12):
        return -1.0
        
    # Wall distance
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    # Pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_wall, min_pair / 2.0)
    
    # If min_pair is 0, points are coincident, radius is 0
    if min_pair < 1e-12:
        return 0.0
        
    return r
