import numpy as np
import time

N = 100

def _compute_radius(centres):
    """
    Compute the maximum common radius for given centres.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if len(centres) == 0:
        return 0.0
    
    # Distance to boundary
    # centres shape (N, 3)
    # boundary distances: 0.5 - |x|, 0.5 + |x| etc.
    # min(0.5 - |c_ik|) for each k
    abs_centres = np.abs(centres)
    dist_to_face = 0.5 - abs_centres
    min_boundary_dist = np.min(dist_to_face)
    
    # Pairwise distances
    if N > 1:
        # Compute pairwise distances efficiently
        # diff[i, j] = centres[i] - centres[j]
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    r = min(min_boundary_dist, min_pair_dist / 2.0)
    return r

def _initial_grid(n):
    """
    Create a 5x4x5 grid of points inside the cube, with jitter.
    """
    nx, ny, nz = 5, 4, 5
    # Ensure nx*ny*nz >= n
    # 5*4*5 = 100 exactly.
    
    # Define grid points
    # x in [-0.5, 0.5], 5 points -> spacing 0.25, centers at -0.375, -0.125, 0.125, 0.375... wait.
    # If we want 5 points in [-0.5, 0.5], we can use linspace(-0.4, 0.4, 5) or similar to keep them away from edges initially.
    # Better: use uniform spacing that maximizes min distance.
    # For 1D interval [-0.5, 0.5] with m points, optimal is equally spaced.
    # x_k = -0.5 + (k+0.5)/m
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Add jitter
    np.random.seed(42)
    jitter = np.random.uniform(-0.02, 0.02, grid.shape)
    grid += jitter
    
    # Clip to ensure inside
    grid = np.clip(grid, -0.45, 0.45)
    
    return grid[:n]

def _energy_delta(centres, i, new_c_i, k=10):
    """
    Calculate the change in energy if point i is moved to new_c_i.
    Energy E = sum_{j < l} d_jl^{-k} + sum_i w_i^{-k}
    Delta E = (sum_{j != i} d'_ij^{-k} + w'_i^{-k}) - (sum_{j != i} d_ij^{-k} + w_i^{-k})
    """
    # Current distances from i to all others
    diffs = centres - centres[i]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    dists[i] = np.inf # Exclude self
    
    # Current wall distance
    abs_c = np.abs(centres[i])
    w_old = 0.5 - np.min(abs_c)
    
    # New distances
    diffs_new = centres - new_c_i
    dists_new = np.sqrt(np.sum(diffs_new ** 2, axis=1))
    dists_new[i] = np.inf
    
    # New wall distance
    abs_new = np.abs(new_c_i)
    w_new = 0.5 - np.min(abs_new)
    
    # Avoid division by zero or very small numbers
    eps = 1e-12
    dists = np.maximum(dists, eps)
    dists_new = np.maximum(dists_new, eps)
    w_old = max(w_old, eps)
    w_new = max(w_new, eps)
    
    e_old = np.sum(dists ** (-k)) + (w_old ** (-k))
    e_new = np.sum(dists_new ** (-k)) + (w_new ** (-k))
    
    return e_new - e_old

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Initialize
    centres = _initial_grid(N).copy()
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    # Pre-compute initial energy components? 
    # We will update energy incrementally.
    # Let's compute the full energy once to have a baseline, though we only need deltas.
    # For SA, we need E_old.
    
    # We can just track the current energy value.
    # Calculate initial energy
    k_power = 10
    def calc_total_energy(c):
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, 1e-12)
        pair_e = np.sum(dists ** (-k_power)) / 2.0 # Divide by 2 because pair counted twice
        
        abs_c = np.abs(c)
        w = 0.5 - np.min(abs_c, axis=1)
        w = np.maximum(w, 1e-12)
        wall_e = np.sum(w ** (-k_power))
        
        return pair_e + wall_e
        
    current_energy = calc_total_energy(centres)
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.9995 # Cooling rate
    T = T_start
    
    np.random.seed(42)
    
    # Main Loop
    while time.time() - start_time < deadline - 2.0:
        if T < T_end:
            break
            
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Propose a move: Gaussian perturbation
        # Scale of perturbation can depend on T
        step_size = 0.05
        delta = np.random.normal(0, step_size, 3)
        
        new_c_i = centres[i] + delta
        
        # Keep inside cube
        # If it goes outside, we can either reject or clip.
        # Clipping changes the effective step. Let's clip to [-0.49, 0.49] to stay strictly inside for safety, 
        # but the energy function handles boundary distance.
        # Actually, if we clip, the "move" is not a simple Gaussian.
        # Better: If new_c_i is outside, reject immediately or reflect?
        # Reflection is tricky. Let's just reject if outside [-0.5, 0.5].
        if np.any(new_c_i < -0.5) or np.any(new_c_i > 0.5):
            # Try a smaller step or just skip
            continue
            
        # Calculate Delta Energy
        dE = _energy_delta(centres, i, new_c_i, k=k_power)
        
        # Acceptance Criterion
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            # Accept
            centres[i] = new_c_i
            current_energy += dE
            
            # Check if this is a better radius
            # We don't want to compute full radius every step (O(N^2)).
            # We can check periodically or if dE is very negative.
            # Let's check every 1000 steps.
            
        # Cool down
        T *= alpha
        
        # Periodic radius check
        # This is expensive, so do it less frequently
        # We can use a counter
        # Since we can't easily access iteration count in this structure without adding it, 
        # let's just check time and maybe do a few checks.
        
        # Actually, computing radius is O(N^2). 
        # We can optimize by only updating the best if the current energy has improved significantly?
        # No, energy is a proxy. 
        # Let's just check every ~1000 iterations.
        # To do this, we need a counter.
        
        # Let's restructure slightly to include a counter.
        
    # The loop above is a bit messy with the counter. Let's rewrite the loop properly.
    
    return best_centres
