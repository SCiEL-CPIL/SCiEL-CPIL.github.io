"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit
    
    # 1. Initialization: Jittered Grid
    # Start with a reasonable spread to avoid initial overlaps
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    positions = grid[:N].copy()
    
    # Add jitter to break symmetry
    positions += np.random.uniform(-0.02, 0.02, size=positions.shape)
    positions = np.clip(positions, -0.49, 0.49)
    
    # Compute initial radius
    current_radius = _compute_radius(positions)
    best_positions = positions.copy()
    best_radius = current_radius
    
    # 2. Simulated Annealing Parameters
    # We want to maximize r. 
    # Objective function to minimize: -r (or 1/r, but -r is simpler for SA acceptance)
    # Temperature starts high enough to allow large moves and escape local minima.
    T = 0.005
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Perturbation size
    # Start with larger moves, reduce over time
    step_size = 0.05
    
    # Precompute indices for faster distance calculation if needed, 
    # but N=200 is small enough for vectorized full pairwise dist in numpy.
    
    steps = 0
    while time.time() < deadline:
        # Generate a new candidate position by perturbing a random subset of points
        # or all points with small noise. 
        # For packing, perturbing all points slightly is often effective.
        # Let's perturb all points with Gaussian noise scaled by step_size.
        
        # To make the search more effective, we can also try "single point" moves 
        # or "pair" moves, but full-system perturbation is simpler and works well 
        # when combined with annealing.
        
        # Actually, a better strategy for packing is often to pick a random point 
        # and move it significantly, or move all points slightly.
        # Let's use a mix: 
        # 80% chance: Move a random single point significantly (exploration)
        # 20% chance: Move all points slightly (exploitation/refinement)
        
        if random.random() < 0.8:
            # Move a random point
            idx = random.randint(0, N - 1)
            new_pos = positions.copy()
            # Large step for single point to escape local traps
            noise = np.random.normal(0, step_size * 2, size=2)
            new_pos[idx] += noise
            # Clamp to box
            new_pos[idx] = np.clip(new_pos[idx], -0.5, 0.5)
        else:
            # Move all points slightly
            new_pos = positions + np.random.normal(0, step_size * 0.1, size=positions.shape)
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
        # Compute new radius
        new_radius = _compute_radius(new_pos)
        
        # Acceptance criterion
        delta_r = new_radius - current_radius
        
        if delta_r > 0:
            # Improvement: always accept
            positions = new_pos
            current_radius = new_radius
            
            # Update best
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
        else:
            # Worse: accept with probability exp(delta_r / T)
            # Note: delta_r is negative.
            # We use -delta_r as the energy difference for minimization of -radius
            # So P = exp(delta_r / T) where delta_r is negative.
            # If T is small, this is close to 0.
            if T > T_min:
                prob = np.exp(delta_r / T)
                if random.random() < prob:
                    positions = new_pos
                    current_radius = new_radius
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Adapt step size: reduce it slowly to refine the solution
        # step_size *= 0.9999
        # Instead, let's keep step_size constant but rely on T for acceptance.
        # Or reduce step size to match the cooling.
        step_size *= 0.9999
        if step_size < 0.001:
            step_size = 0.001
            
        steps += 1
        
        # Safety check
        if steps % 1000 == 0:
             if time.time() > deadline - 5.0:
                 break

    return best_positions

def _compute_radius(positions):
    """Compute the maximum possible radius for given positions."""
    if positions.shape[0] == 0:
        return 0.0
    
    # Min distance to walls
    # Square is [-0.5, 0.5]^2
    x, y = positions[:, 0], positions[:, 1]
    
    # Distance from each center to the nearest wall
    # d_x = min(0.5 - x, 0.5 + x)
    # d_y = min(0.5 - y, 0.5 + y)
    # d_wall_i = min(d_x_i, d_y_i)
    # r_wall = min_i d_wall_i
    
    d_x = np.minimum(0.5 - x, 0.5 + x)
    d_y = np.minimum(0.5 - y, 0.5 + y)
    d_wall = np.minimum(d_x, d_y)
    min_dist_wall = np.min(d_wall)
    
    # Min distance between centers
    # Vectorized pairwise distance
    # diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # dist_sq = np.sum(diff**2, axis=-1)
    # np.fill_diagonal(dist_sq, np.inf)
    # min_dist_center = np.sqrt(np.min(dist_sq))
    
    # For N=200, this is 40,000 ops. Very fast in numpy.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_center = np.sqrt(np.min(dist_sq))
    
    # Radius is half the minimum of wall distance and center-center distance
    r = min(min_dist_wall, min_dist_center / 2.0)
    
    if r < 0:
        return 0.0
    return r
# EVOLVE-BLOCK-END
