"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Optimized repulsion-based local search for packing 50 spheres in a unit cube.
    Uses vectorized operations for speed, multiple initializations, and adaptive
    step sizes to maximize the minimum distance between spheres and boundaries.
    """
    rng = np.random.default_rng(42)
    
    def compute_radius(centers):
        """Compute the maximum possible radius for the given centers."""
        # Pairwise distances
        # Using broadcasting for speed
        # Shape (N, 1, 3) - (N, N, 3) -> (N, N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_pair_dist = np.min(dist_matrix)
        
        # Distance to boundaries
        # Centers are in [-0.5, 0.5]
        # Distance to wall is 0.5 - |coord|
        abs_centers = np.abs(centers)
        dist_to_walls = 0.5 - np.max(abs_centers, axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        min_dist = min(min_pair_dist, min_wall_dist)
        return min_dist / 2.0

    def optimize(centers, iterations=500, start_step=0.05):
        """
        Perform local optimization using repulsive forces.
        """
        centers = centers.copy()
        step_size = start_step
        
        for i in range(iterations):
            # 1. Calculate Sphere-Sphere Forces
            # diff[i, j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            dist_matrix[dist_matrix < 1e-10] = 1e-10
            # Set diagonal to 1 to avoid self-force contribution (will be zeroed out by diff)
            np.fill_diagonal(dist_matrix, 1.0)
            
            # Force magnitude: 1 / dist^2
            # We want the force vector to be diff / dist * (1/dist^2) = diff / dist^3
            force_inv_dist3 = 1.0 / (dist_matrix ** 3)
            
            # Sum forces for each sphere
            # F_i = sum_j (diff[i,j] * force_inv_dist3[i,j])
            forces = np.sum(diff * force_inv_dist3[:, :, np.newaxis], axis=1)
            
            # 2. Calculate Wall Forces
            # Push away from walls if close
            # We want to maximize distance to wall, so if x > 0, push left (negative force)
            # Force ~ 1 / d_wall^2
            for k in range(3):
                coords = centers[:, k]
                # Distance to positive wall
                d_pos = 0.5 - coords
                # Distance to negative wall
                d_neg = coords + 0.5
                
                # Only apply force if close to wall (e.g., d < 0.2)
                # To make it smooth, we can apply it everywhere but scaled
                # Let's use a soft repulsion from the walls
                # If coord > 0, wall is at 0.5. Force is negative.
                # If coord < 0, wall is at -0.5. Force is positive.
                
                # Calculate force from positive wall
                mask_pos = coords > 0
                d_pos_safe = np.where(d_pos < 1e-6, 1e-6, d_pos)
                f_pos = np.zeros_like(coords)
                # We want the force to be strong when close.
                # Let's use 1/d^2
                f_pos[mask_pos] = -1.0 / (d_pos_safe[mask_pos] ** 2)
                
                # Calculate force from negative wall
                mask_neg = coords < 0
                d_neg_safe = np.where(d_neg < 1e-6, 1e-6, d_neg)
                f_neg = np.zeros_like(coords)
                f_neg[mask_neg] = 1.0 / (d_neg_safe[mask_neg] ** 2)
                
                forces[:, k] += f_pos + f_neg
            
            # Normalize forces to prevent explosion
            force_norms = np.linalg.norm(forces, axis=1)
            # Limit max force magnitude
            max_force = 1.0
            for i in range(N):
                if force_norms[i] > max_force:
                    forces[i] *= max_force / force_norms[i]
            
            # Update centers
            centers += step_size * forces
            centers = np.clip(centers, -0.499, 0.499)
            
            # Decay step size
            step_size *= 0.995
            
            # If step size is too small, break
            if step_size < 1e-5:
                break
                
        return centers

    best_centers = None
    best_radius = -np.inf
    
    # Define initializations
    initializations = []
    
    # 1. Grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initializations.append(grid[:N].copy())
    
    # 2. Random
    initializations.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. Fibonacci-like / Structured Random
    # Generate points on a sphere and project? No, just use a stratified random
    # Divide cube into cells, pick one random point per cell if possible
    cells = np.array([(x, y, z) for z in range(4) for y in range(4) for x in range(4)])
    # We have 64 cells, need 50 points
    # Shuffle cells and pick 50
    shuffled_cells = rng.permutation(cells)
    stratified = []
    for cell in shuffled_cells[:N]:
        # Pick a random point within the cell
        x = cell[0] / 4.0 + rng.uniform(0, 1/4.0) - 0.5
        y = cell[1] / 4.0 + rng.uniform(0, 1/4.0) - 0.5
        z = cell[2] / 4.0 + rng.uniform(0, 1/4.0) - 0.5
        stratified.append([x, y, z])
    initializations.append(np.array(stratified))
    
    # 4. Another random start
    initializations.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 5. Grid with more points, then subsample
    m2 = int(np.ceil(N ** (1.0 / 3.0)) + 2)
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
    # Randomly select N points from grid2
    idx = rng.choice(len(grid2), N, replace=False)
    initializations.append(grid2[idx].copy())
    
    # Optimize each initialization
    for init in initializations:
        # Run optimization
        optimized = optimize(init, iterations=800, start_step=0.05)
        radius = compute_radius(optimized)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
            
        # Check time
        # If we are running low on time, break
        # (We don't have a strict timer here, but 5 inits * 800 iters * vectorized ops is fast)
        
    # Final polish on the best center
    # Run a few more iterations with smaller step
    if best_centers is not None:
        final_centers = optimize(best_centers, iterations=200, start_step=0.01)
        final_radius = compute_radius(final_centers)
        if final_radius > best_radius:
            best_centers = final_centers
            best_radius = final_radius
            
    return best_centers
# EVOLVE-BLOCK-END
