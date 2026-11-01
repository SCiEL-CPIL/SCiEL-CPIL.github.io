"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
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

def _local_refine(c, max_iters=20000, tol=1e-10):
    """
    Local refinement: iteratively push apart the closest pair of spheres
    and pull in the sphere closest to a wall. This is much faster than
    scipy optimizers for the max-min landscape.
    """
    c = c.copy()
    N_local = len(c)
    
    for it in range(max_iters):
        # Find closest pair
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        
        d = dists[i, j]
        
        # Find closest to wall
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k = min_wall_idx[0]
        axis = min_wall_idx[1]
        
        # Determine which constraint is binding
        min_pair_half = d / 2.0
        min_wall_val = wall_dists[k, axis]
        
        if min_pair_half <= min_wall_val:
            # Binding constraint is pair distance: push i and j apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                # Coincident points: use random direction
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Push apart by a small step, but respect walls
            step = 0.001
            new_ci = c[i] + direction * step
            new_cj = c[j] - direction * step
            
            # Check wall constraints for new positions
            # For each coordinate, ensure within [-0.5, 0.5]
            # If pushing would violate a wall, reduce step
            for coord in range(3):
                if new_ci[coord] > 0.5:
                    step = min(step, (0.5 - c[i][coord]) / max(direction[coord], 1e-15))
                elif new_ci[coord] < -0.5:
                    step = min(step, (-0.5 - c[i][coord]) / min(direction[coord], -1e-15))
                if new_cj[coord] > 0.5:
                    step = min(step, (0.5 - c[j][coord]) / max(-direction[coord], 1e-15))
                elif new_cj[coord] < -0.5:
                    step = min(step, (-0.5 - c[j][coord]) / min(-direction[coord], -1e-15))
            
            if step > 1e-12:
                c[i] += direction * step
                c[j] -= direction * step
        else:
            # Binding constraint is wall: pull k inward along axis
            # Determine direction: if c[k,axis] > 0, pull negative; else pull positive
            if c[k, axis] > 0:
                direction = -1.0
            else:
                direction = 1.0
            
            step = 0.001
            new_ck = c[k].copy()
            new_ck[axis] += direction * step
            
            # Check if this violates any pair constraint
            # Simple approach: just do it, next iteration will fix pair issues
            c[k] = new_ck
    
    return c

def _adaptive_refine(c, time_limit, seed=0):
    """
    Adaptive step size refinement with occasional random escapes.
    Uses a decaying step size and random perturbations to escape local minima.
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
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(200000):
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
                    max_step_j = min(max_step_j, space / direction[coord])
            
            actual_step = min(step, max_step_i, max_step_j)
            if actual_step > min_step:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        else:
            # Pull k inward along axis
            if c[k, axis] > 0:
                direction = -1.0
            else:
                direction = 1.0
            
            # Calculate max allowed step to stay in bounds
            if direction > 0:
                max_step = 0.5 - c[k, axis]
            else:
                max_step = c[k, axis] + 0.5
            
            actual_step = min(step, max_step)
            if actual_step > min_step:
                c[k, axis] += direction * actual_step
        
        # Decay step size
        step *= decay
        
        # Occasional random escape to avoid local minima
        if it % 500 == 0:
            # Small random perturbation
            perturbation = rng.randn(N_local, 3) * 0.001
            c = c + perturbation
            c = np.clip(c, -0.4999, 0.4999)
        
        # Track best configuration
        current_r = _compute_radius(c)
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Get initial configuration
    c0 = _initial_configuration()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Run local refinement
    c_refined = _local_refine(c0, max_iters=50000)
    r_refined = _compute_radius(c_refined)
    
    if r_refined > best_r:
        best_r = r_refined
        best_c = c_refined.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_c
    
    # Try adaptive refinement with different seeds
    for seed in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Use remaining time for this refinement
        time_for_refine = min(20.0, deadline - 10 - elapsed)
        
        c_adapt, r_adapt = _adaptive_refine(c0, time_for_refine, seed=seed)
        
        if r_adapt > best_r:
            best_r = r_adapt
            best_c = c_adapt.copy()
    
    # Try a perturbed version for diversity
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        np.random.seed(42)
        noise = np.random.normal(0, 0.005, c0.shape)
        c_perturbed = np.clip(c0 + noise, -0.495, 0.495)
        
        time_for_pert = min(15.0, deadline - 10 - elapsed)
        c_pert_refined, r_pert = _adaptive_refine(c_perturbed, time_for_pert, seed=100)
        
        if r_pert > best_r:
            best_r = r_pert
            best_c = c_pert_refined.copy()
    
    # Final verification
    if not np.all(np.isfinite(best_c)):
        return c0
    
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
# EVOLVE-BLOCK-END
