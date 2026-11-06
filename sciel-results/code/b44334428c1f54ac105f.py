"""
Rewritten program for packing 200 equal circles in a unit square.
Uses a combination of global repulsion, bottleneck-driven local optimization, and polishing.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

N = 200
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _remaining_time(start_time):
    return TIME_LIMIT - (time.time() - start_time)

def _compute_radius(centres):
    """Compute the largest common radius for the given centres."""
    # Distance to walls
    # centres are in [-0.5, 0.5]^2
    dist_to_walls = np.minimum(
        np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
        np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
    )
    min_wall = np.min(dist_to_walls)
    
    # Pairwise distances
    if len(centres) < 2:
        min_pair = np.inf
    else:
        dists = pdist(centres)
        min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _pdist_index_to_pair(i, n):
    """Convert pdist linear index to pair (i, j) with i < j."""
    # pdist index k corresponds to pair (i, j) where:
    # k = i*n - i*(i+1)/2 + (j - i - 1)
    # We can solve for i and j, or just use a lookup if N is small enough.
    # For N=200, N*(N-1)/2 = 19900. A lookup table is feasible and fast.
    # However, creating it every time is slow. We can derive it.
    # Alternatively, since we often just need the min, we can use np.argmin on pdist 
    # and then map.
    pass

def _get_min_pair(centres):
    """Return the indices of the closest pair and the distance."""
    if len(centres) < 2:
        return 0, 1, np.inf
    dists = pdist(centres)
    min_idx = np.argmin(dists)
    # Convert linear index to pair indices
    # pdist stores distances for pairs (i, j) with i < j in row-major order of the upper triangle.
    # Index mapping:
    # k = i*N - i*(i+1)//2 + (j - i - 1)
    # We can precompute the mapping for N=200 once.
    n = len(centres)
    # Precomputed mapping for N=200 is overkill if done every call, but we can compute on fly.
    # Actually, for N=200, computing the mapping is trivial.
    # Let's just use the squareform approach for simplicity in local opt, but for global min search pdist is faster.
    # To map min_idx back to (i, j):
    # We can iterate or use a formula.
    # Formula: i = floor((sqrt(8*k + 1) - 1) / 2) is not quite right for this specific layout.
    # Let's just use a helper that generates the pairs once and caches it? 
    # Or simpler: since N is fixed at 200, we can just compute the full distance matrix for local ops 
    # and use pdist for global min.
    
    # For now, let's just return the min distance. For local optimization, we will compute the matrix.
    return min_idx, np.min(dists)

def _initial_grid(n):
    """Generate an initial grid configuration."""
    m = int(np.ceil(np.sqrt(n)))
    # Adjust grid to fit n points better
    # If n is not a perfect square, we might have empty spots.
    # Standard approach: use a grid of size m x m, take first n points.
    # Better: distribute points evenly.
    
    # Simple grid
    x = np.linspace(-0.5, 0.5, m)
    y = np.linspace(-0.5, 0.5, m)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return points[:n]

def _optimize_global(centres, start_time, time_limit):
    """Global repulsion optimization."""
    centres = centres.copy()
    n = len(centres)
    best_r = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Parameters
    lr = 0.005
    decay = 0.9995
    steps = 5000
    
    for step in range(steps):
        if _remaining_time(start_time) < time_limit:
            break
            
        # Compute forces
        forces = np.zeros_like(centres)
        
        # Pairwise repulsion
        # Use pdist for efficiency in finding close pairs, but we need forces for all.
        # For N=200, full matrix is fine (40k elements).
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1.0) # Avoid div by zero, set to 1 so force is 0
        
        # Repulsion force: 1/d^2
        # Only apply if dist < some threshold to save computation? No, 200 is small.
        repulsion = 1.0 / (dist**2)
        np.fill_diagonal(repulsion, 0)
        
        forces += np.sum(repulsion[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces
        # Push away from walls
        # d_x = 0.5 - |x|
        dx = 0.5 - np.abs(centres[:, 0])
        dy = 0.5 - np.abs(centres[:, 1])
        
        # Force is strong when close to wall
        # Use 1/d^2
        wall_strength = 0.5
        forces[:, 0] += wall_strength * np.sign(centres[:, 0]) / (dx**2 + 1e-6)
        forces[:, 1] += wall_strength * np.sign(centres[:, 1]) / (dy**2 + 1e-6)
        
        # Update
        centres += lr * forces
        
        # Clamp
        centres[:, 0] = np.clip(centres[:, 0], -0.5, 0.5)
        centres[:, 1] = np.clip(centres[:, 1], -0.5, 0.5)
        
        # Check radius
        r = _compute_radius(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
            
        lr *= decay
        
    return best_centres, best_r

def _local_bottleneck_opt(centres, start_time, time_limit):
    """
    Local optimization focusing on the bottleneck (closest pair and wall contact).
    """
    centres = centres.copy()
    n = len(centres)
    best_r = _compute_radius(centres)
    best_centres = centres.copy()
    
    steps = 2000
    lr = 0.01
    
    for step in range(steps):
        if _remaining_time(start_time) < time_limit:
            break
            
        # Compute distances
        dists = squareform(pdist(centres))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        min_dist_idx = np.argmin(dists)
        i, j = np.unravel_index(min_dist_idx, (n, n))
        
        # Find wall bottleneck
        wall_dists = np.minimum(
            np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
        )
        wall_idx = np.argmin(wall_dists)
        
        current_r = min(np.min(dists)/2.0, np.min(wall_dists))
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
            
        # Move i and j apart
        vec = centres[i] - centres[j]
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            unit_vec = vec / norm
            # Push i in direction of unit_vec, j in opposite
            centres[i] += lr * unit_vec
            centres[j] -= lr * unit_vec
            
        # Move wall_idx away from wall
        # Determine which wall
        x, y = centres[wall_idx]
        if abs(x + 0.5) < abs(x - 0.5):
            centres[wall_idx, 0] += lr
        else:
            centres[wall_idx, 0] -= lr
        if abs(y + 0.5) < abs(y - 0.5):
            centres[wall_idx, 1] += lr
        else:
            centres[wall_idx, 1] -= lr
            
        # Clamp
        centres[:, 0] = np.clip(centres[:, 0], -0.5, 0.5)
        centres[:, 1] = np.clip(centres[:, 1], -0.5, 0.5)
        
        # Decay learning rate
        lr *= 0.999

    return best_centres, best_r

def _polish(centres, start_time, time_limit):
    """Fine-tuning with very small steps."""
    centres = centres.copy()
    best_r = _compute_radius(centres)
    best_centres = centres.copy()
    
    steps = 500
    lr = 0.001
    
    for step in range(steps):
        if _remaining_time(start_time) < time_limit:
            break
            
        dists = squareform(pdist(centres))
        np.fill_diagonal(dists, np.inf)
        
        min_dist_idx = np.argmin(dists)
        i, j = np.unravel_index(min_dist_idx, (len(centres), len(centres)))
        
        vec = centres[i] - centres[j]
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            unit_vec = vec / norm
            centres[i] += lr * unit_vec
            centres[j] -= lr * unit_vec
            
        # Wall polish
        wall_dists = np.minimum(
            np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
        )
        wall_idx = np.argmin(wall_dists)
        x, y = centres[wall_idx]
        if abs(x + 0.5) < abs(x - 0.5):
            centres[wall_idx, 0] += lr
        else:
            centres[wall_idx, 0] -= lr
        if abs(y + 0.5) < abs(y - 0.5):
            centres[wall_idx, 1] += lr
        else:
            centres[wall_idx, 1] -= lr
            
        centres[:, 0] = np.clip(centres[:, 0], -0.5, 0.5)
        centres[:, 1] = np.clip(centres[:, 1], -0.5, 0.5)
        
        r = _compute_radius(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
            
    return best_centres, best_r

def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    start_time = time.time()
    
    # Initial grid
    centres = _initial_grid(N)
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    np.random.seed(42)
    
    # Main loop: multiple restarts
    # Each restart: Global Opt -> Local Opt -> Polish
    # We allocate time based on remaining budget.
    
    num_restarts = 10 # Maximum
    
    for restart in range(num_restarts):
        if _remaining_time(start_time) < 10.0: # Need at least 10s for one cycle
            break
            
        # Perturb initial config for restarts > 0
        if restart == 0:
            curr_centres = _initial_grid(N)
        else:
            # Perturb best found so far or initial grid?
            # Perturbing best found is usually better for refinement, but might get stuck.
            # Perturbing initial grid explores new basins.
            # Let's mix: 50% perturb best, 50% perturb initial.
            if np.random.rand() < 0.5:
                curr_centres = best_centres + np.random.randn(N, 2) * 0.01
            else:
                curr_centres = _initial_grid(N) + np.random.randn(N, 2) * 0.05
                
            # Ensure within bounds
            curr_centres[:, 0] = np.clip(curr_centres[:, 0], -0.5, 0.5)
            curr_centres[:, 1] = np.clip(curr_centres[:, 1], -0.5, 0.5)
            
        # 1. Global Optimization
        # Allocate 30% of remaining time (min 2s)
        t_global = max(2.0, _remaining_time(start_time) * 0.3)
        curr_centres, r_global = _optimize_global(curr_centres, start_time, t_global)
        
        if _remaining_time(start_time) < 5.0:
            break
            
        # 2. Local Bottleneck Optimization
        t_local = max(2.0, _remaining_time(start_time) * 0.3)
        curr_centres, r_local = _local_bottleneck_opt(curr_centres, start_time, t_local)
        
        if _remaining_time(start_time) < 2.0:
            break
            
        # 3. Polish
        t_polish = max(1.0, _remaining_time(start_time) * 0.2)
        curr_centres, r_final = _polish(curr_centres, start_time, t_polish)
        
        if r_final > best_r:
            best_r = r_final
            best_centres = curr_centres.copy()
            
    return best_centres
