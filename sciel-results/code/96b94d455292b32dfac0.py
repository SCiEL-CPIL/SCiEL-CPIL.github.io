"""Improved packing of 50 equal circles in a unit square via Simulated Annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances
    if len(centres) < 2:
        min_pair = np.inf
    else:
        # Vectorized pairwise distances
        # diff[i, j] = centres[i] - centres[j]
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _construct_grid():
    """Create a grid-based initial configuration."""
    # 7x7 grid is 49. 8x7 is 56.
    # Let's use a 7x8 grid and take 50.
    m_x = 7
    m_y = 8
    ticks_x = np.linspace(-0.5 + 1/(2*m_x), 0.5 - 1/(2*m_x), m_x)
    ticks_y = np.linspace(-0.5 + 1/(2*m_y), 0.5 - 1/(2*m_y), m_y)
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N]

def _sa_optimize(centres, t_start=0.1, t_end=0.0001, n_steps=20000, seed=42):
    """
    Simulated Annealing to maximize the minimum radius.
    """
    np.random.seed(seed)
    curr_centres = centres.copy()
    curr_r = _radius_score(curr_centres)
    
    best_centres = curr_centres.copy()
    best_r = curr_r
    
    # Precompute indices for efficiency?
    # For now, just vectorized full recompute.
    
    for i in range(n_steps):
        # Cooling schedule: exponential
        t = t_start * (t_end / t_start) ** (i / n_steps)
        
        # Move one random circle
        idx = np.random.randint(N)
        # Step size proportional to temperature?
        step_size = 0.05 * (t / t_start) ** 0.5 + 0.001
        delta = np.random.normal(0, step_size, 2)
        
        new_centres = curr_centres.copy()
        new_centres[idx] += delta
        
        # Clip to square to avoid huge boundary violations
        new_centres[idx] = np.clip(new_centres[idx], -0.5, 0.5)
        
        new_r = _radius_score(new_centres)
        
        # Acceptance criterion
        # We want to maximize r.
        # Delta E = curr_r - new_r. If new_r > curr_r, Delta E < 0 (good).
        delta_e = curr_r - new_r
        
        if delta_e < 0:
            # Improvement
            curr_centres = new_centres
            curr_r = new_r
            if curr_r > best_r:
                best_r = curr_r
                best_centres = curr_centres.copy()
        else:
            # Worse, accept with probability exp(-delta_e / t)
            if np.random.random() < np.exp(-delta_e / t):
                curr_centres = new_centres
                curr_r = new_r
                
    return best_centres, best_r

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    best_centres = None
    best_score = 0.0
    
    start_time = time.time()
    time_limit = 100 # Leave buffer for overhead
    
    # Multi-start SA
    # 1. Grid
    # 2. Perturbed Grids
    # 3. Random
    
    starts = []
    
    # Grid
    grid = _construct_grid()
    starts.append(grid)
    
    # Perturbed Grids
    np.random.seed(0)
    for i in range(10):
        noise = np.random.normal(0, 0.1, size=(N, 2))
        perturbed = grid + noise
        perturbed = np.clip(perturbed, -0.5, 0.5)
        starts.append(perturbed)
        
    # Random
    for i in range(5):
        random_pts = np.random.uniform(-0.5, 0.5, size=(N, 2))
        starts.append(random_pts)
        
    # Run SA on each start
    # We need to allocate time wisely.
    # Let's run 5000 steps per start initially, then refine the best.
    
    n_starts = len(starts)
    steps_per_start = 5000
    
    for i, start in enumerate(starts):
        if time.time() - start_time > time_limit - 10:
            break
            
        s = i + 1
        # Adjust steps based on time
        # If we have more time, use more steps.
        # For now, fixed steps.
        
        opt_centres, opt_score = _sa_optimize(start, n_steps=steps_per_start, seed=i)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centres = opt_centres
            
    # Refine the best result with longer SA run
    if best_centres is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            # Scale steps by time
            # ~5000 steps took T seconds.
            # Estimate steps for remaining time.
            # Rough estimate: 5000 steps is fast, maybe 0.5s?
            # Let's just do a large number of steps.
            opt_centres, opt_score = _sa_optimize(best_centres, t_start=0.01, t_end=0.00001, n_steps=50000, seed=999)
            if opt_score > best_score:
                best_score = opt_score
                best_centres = opt_centres
                
    if best_centres is None:
        return _construct_grid()
        
    return best_centres
# EVOLVE-BLOCK-END
