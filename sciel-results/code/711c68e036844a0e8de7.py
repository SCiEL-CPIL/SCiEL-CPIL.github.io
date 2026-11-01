"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if len(c) == 0:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _initial_configuration():
    """Create a 5x5x2 grid initialization that exactly fits 50 spheres."""
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    
    return np.array(grid)

def _direct_refine(c, time_limit, seed=0):
    """
    Direct repulsion refinement with adaptive step size.
    Iteratively pushes apart the closest pair or pulls in the sphere closest to a wall.
    """
    c = c.copy()
    N_local = len(c)
    start_time = time.time()
    
    # Adaptive step size parameters
    step = 0.01
    decay = 0.995
    min_step = 1e-6
    
    # Random state for tie-breaking or small perturbations if needed
    rng = np.random.RandomState(seed)
    
    for it in range(200000): # High cap, but time-limited
        if time.time() - start_time > time_limit:
            break
            
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
            
        d_pair = dists[i, j]
        
        # Compute wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, axis = min_wall_idx
        d_wall = wall_dists[k, axis]
        
        # Determine which constraint is binding
        # We want to maximize r = min(d_pair/2, d_wall)
        # If d_pair/2 < d_wall, the pair is limiting. Push them apart.
        # If d_wall < d_pair/2, the wall is limiting. Pull k in.
        
        if d_pair / 2.0 < d_wall:
            # Push i and j apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = rng.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Try to move
            # We want to increase d_pair. 
            # New dist approx d_pair + 2*step
            # But we must respect walls.
            
            # Calculate max allowed step for i and j to stay in bounds
            max_step_i = np.inf
            max_step_j = np.inf
            
            for coord in range(3):
                # For i moving in +direction
                if direction[coord] > 0:
                    space = 0.5 - c[i][coord]
                    if space < 0: space = 0
                    max_step_i = min(max_step_i, space / direction[coord])
                elif direction[coord] < 0:
                    space = c[i][coord] + 0.5
                    if space < 0: space = 0
                    max_step_i = min(max_step_i, space / (-direction[coord]))
                
                # For j moving in -direction
                if -direction[coord] > 0:
                    space = 0.5 - c[j][coord]
                    if space < 0: space = 0
                    max_step_j = min(max_step_j, space / (-direction[coord]))
                elif -direction[coord] < 0:
                    space = c[j][coord] + 0.5
                    if space < 0: space = 0
                    max_step_j = min(max_step_j, space / (direction[coord]))
            
            actual_step = min(step, max_step_i, max_step_j)
            
            if actual_step > 1e-12:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        else:
            # Pull k inward along axis
            # Determine direction: if c[k,axis] > 0, move -1; else +1
            if c[k, axis] > 0:
                dir_val = -1.0
            else:
                dir_val = 1.0
            
            # Max step to wall is d_wall (we are already at the min wall dist)
            # But we also don't want to violate pair constraints too badly, 
            # though the next iteration will fix that.
            # Actually, if we pull in, we might get closer to other spheres.
            # Let's just move a small step.
            
            actual_step = min(step, d_wall + 1e-9) # Can't go past wall
            
            if actual_step > 1e-12:
                c[k, axis] += dir_val * actual_step
        
        # Decay step size
        step = max(min_step, step * decay)
        
        # Occasionally add small random noise to escape local minima
        if it % 1000 == 0 and step > 1e-4:
             noise = rng.normal(0, step * 0.1, c.shape)
             c = np.clip(c + noise, -0.5, 0.5)
            
    return c

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Get initial configuration
    c0 = _initial_configuration()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Strategy: Run direct refinement on the base grid, then on perturbed versions
    # Time budget allocation:
    # 1. Base grid: 30s
    # 2. Perturbation 1: 30s
    # 3. Perturbation 2: 30s
    # 4. Final Polish: 15s
    
    allocations = [30.0, 30.0, 30.0, 15.0]
    seeds = [42, 123, 456, 789]
    
    current_c = c0.copy()
    
    for idx, alloc in enumerate(allocations):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining <= 0:
            break
        
        # Determine time for this step
        t_limit = min(alloc, remaining - 1.0) # Keep 1s buffer
        
        if t_limit <= 0:
            break
            
        # Prepare input for this refinement
        if idx == 0:
            c_input = c0.copy()
        else:
            # Perturb the best found so far
            rng = np.random.RandomState(seeds[idx])
            noise_scale = 0.05 / (idx + 1) # Decreasing noise
            noise = rng.normal(0, noise_scale, current_c.shape)
            c_input = np.clip(current_c + noise, -0.49, 0.49)
            
        # Run refinement
        c_refined = _direct_refine(c_input, t_limit, seed=seeds[idx])
        
        # Evaluate
        r_refined = _compute_radius(c_refined)
        
        if r_refined > best_r:
            best_r = r_refined
            best_c = c_refined.copy()
        
        # Update current_c for next perturbation to be based on the best
        current_c = best_c.copy()
        
    # Final check and clipping
    if not np.all(np.isfinite(best_c)):
        return c0
        
    # Ensure strictly inside
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
# EVOLVE-BLOCK-END
