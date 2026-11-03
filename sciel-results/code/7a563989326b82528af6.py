"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    if len(pts) < 2:
        return 0.5
    # Distance between all pairs
    n = len(pts)
    min_dist = np.inf
    # Vectorized pairwise distance
    # pts shape (N, 3)
    # diff shape (N, N, 3)
    # To save memory for N=50, we can do it in chunks or just full since 50*50*3 is small
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Exclude diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to walls
    min_wall = np.inf
    for k in range(3):
        d = 0.5 - np.abs(pts[:, k])
        min_wall = min(min_wall, np.min(d))
    
    return min(min_dist / 2.0, min_wall)

def _relax(pts, time_limit, initial_temp=0.1, cooling=0.95):
    """
    Vectorized repulsive force relaxation.
    Force on i from j is proportional to (r_ij - target_r) / r_ij^3 * (p_i - p_j)
    Actually, to maximize min distance, we want to push apart if d < target.
    A common approach is to minimize sum of exp(-alpha * d_ij) or similar.
    Here we use a direct repulsion force: F_ij = k * (1/d_ij^2) if d_ij < threshold.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Parameters for relaxation
    # We want to push spheres apart.
    # Force magnitude: 1/d^2 is strong.
    # Let's use a "soft" repulsion that is active only when close.
    # Or simply: F = (p_i - p_j) / d^3 * (target - d) if d < target else 0.
    # But target is unknown. 
    # Better: minimize sum_{i<j} exp(-beta * d_ij). Gradient is proportional to (p_i - p_j) * exp(-beta * d_ij).
    # This is smooth and encourages separation.
    
    beta = 10.0 # Stiffness
    lr = 0.005  # Learning rate
    
    while time.time() - start_time < time_limit:
        # Compute forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
        dists = np.linalg.norm(diff, axis=2) # (N, N)
        np.fill_diagonal(dists, 1.0) # Avoid div by zero
        
        # Force vector: sum_j (p_i - p_j) * w(d_ij)
        # w(d) = exp(-beta * d)
        weights = np.exp(-beta * dists)
        np.fill_diagonal(weights, 0.0)
        
        # Forces[i] = sum_j weights[i,j] * diff[i,j]
        forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces: push away from walls
        # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
        # We don't know r exactly, but we can use a proxy or just clamp.
        # Let's add a wall repulsion that activates near boundaries.
        # Wall force is independent of other spheres.
        # For each dimension k:
        # If pts[i,k] > 0.4: force_k -= (pts[i,k] - 0.4) * wall_strength
        # If pts[i,k] < -0.4: force_k += (-0.4 - pts[i,k]) * wall_strength
        wall_strength = 5.0
        for k in range(3):
            # Upper wall
            mask_upper = pts[:, k] > 0.4
            if np.any(mask_upper):
                forces[mask_upper, k] -= (pts[mask_upper, k] - 0.4) * wall_strength
            # Lower wall
            mask_lower = pts[:, k] < -0.4
            if np.any(mask_lower):
                forces[mask_lower, k] -= (pts[mask_lower, k] + 0.4) * wall_strength
        
        # Update positions
        pts += lr * forces
        
        # Project back to cube
        pts = np.clip(pts, -0.5, 0.5)
        
        # Compute current radius
        current_r = _compute_radius(pts)
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
        
        # Adaptive learning rate / cooling
        # If we are making progress, keep lr. If not, decrease.
        # Simple heuristic: decrease lr slightly every iteration
        lr *= 0.999
        if lr < 1e-6:
            break
            
    return best_pts, best_r

def _generate_random_init(seed):
    """Generate a random initial configuration with some structure."""
    rng = np.random.RandomState(seed)
    # Start with a grid and add noise
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    # Add random noise
    noise_scale = 0.05
    pts += rng.uniform(-noise_scale, noise_scale, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def construct():
    start_time = time.time()
    time_limit = 120.0 # Safety margin from 150s kill
    
    best_pts = None
    best_r = -1.0
    
    # Strategy: Multiple restarts with different random seeds and relaxation parameters
    # 1. Start with a good structured init (grid + noise)
    # 2. Relax
    # 3. If time remains, try another restart with different noise or different beta
    
    seeds = [42, 123, 999, 2024, 31415]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        pts = _generate_random_init(seed)
        
        # Run relaxation
        remaining = time_limit - (time.time() - start_time) - 2.0
        if remaining <= 0:
            continue
            
        # Split time among restarts? Or just run one long relaxation?
        # Relaxation is vectorized and fast. We can run it for a fixed time slice.
        # Let's give each restart a chunk of time, e.g., 20s each.
        chunk_time = min(remaining, 20.0)
        
        pts_opt, r = _relax(pts, chunk_time)
        
        if r > best_r:
            best_r = r
            best_pts = pts_opt
            
    if best_pts is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
        best_r = _compute_radius(best_pts)
        
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    final_r = _compute_radius(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
