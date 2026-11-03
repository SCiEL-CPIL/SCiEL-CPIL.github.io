"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Using broadcasting for pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def _local_optimize(centers, max_iter=2000, step_size=0.01):
    """Simple gradient-free local optimization (hill climbing)."""
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    
    # Precompute pairs for speed
    n = len(current_centers)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    for it in range(max_iter):
        improved = False
        # Try moving each point in each direction
        # To save time, we can be more selective or use a smaller step
        for i in range(n):
            for d in range(3):
                for sign in [-1, 1]:
                    new_centers = current_centers.copy()
                    new_centers[i, d] += sign * step_size
                    
                    # Check validity (inside cube)
                    if np.any(np.abs(new_centers) > 0.5):
                        continue
                        
                    new_radius = _compute_radius(new_centers)
                    if new_radius > current_radius + 1e-10:
                        current_centers = new_centers
                        current_radius = new_radius
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try smaller step or break
            # For simplicity, break
            break
            
    return current_centers, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial: Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Optimization loop
    # We will do local optimization from the grid start
    # Then maybe random restarts if time permits? 
    # Given 150s limit, local opt from grid might be slow if naive.
    # Let's implement a slightly smarter local opt:
    # Gradient ascent on r? No, r is min of distances.
    # Let's stick to hill climbing but vectorize the check if possible.
    # The current _local_optimize is O(N^3) per step roughly (N points * 6 dirs * O(N^2) check).
    # N=100, N^2=10000. 600 checks * 10000 = 6M ops per iter. 
    # 2000 iters = 12B ops. Too slow in Python.
    
    # Let's optimize _local_optimize to be faster or reduce iterations.
    # Better: use scipy.optimize.minimize on a smooth surrogate?
    # Objective: maximize r.
    # Constraint: d_ij >= 2r, |c_ik| <= 0.5 - r.
    # This is non-smooth.
    
    # Alternative: Fixed r, move points to increase min distance?
    # Or: Minimize sum of exp(-k * d_ij) + sum of exp(-k * (0.5 - |c_ik|))?
    
    # Let's try a simpler approach: 
    # 1. Start with grid.
    # 2. Use scipy.optimize.minimize to minimize a penalty function.
    # Penalty = sum( max(0, 2r - d_ij)^2 ) + sum( max(0, |c_ik| - (0.5 - r))^2 )
    # This is non-differentiable at 0.
    # Use smooth approximation: softplus.
    
    # Let's try a different strategy: 
    # Just do a few random perturbations and local polish using a faster method.
    # Or, since the grid is symmetric, maybe just adjust the grid spacing?
    # No, the optimal packing is likely not a simple grid.
    
    # Let's implement a faster local optimizer using numpy vectorization for the radius calculation.
    # The bottleneck in _local_optimize is _compute_radius.
    
    def fast_radius(c):
        # c is (N, 3)
        wall = 0.5 - np.abs(c)
        min_wall = np.min(wall)
        
        # Pairwise
        # diff = c[:, None, :] - c[None, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # This allocates (100, 100, 3) and (100, 100).
        # Can we do it faster?
        # dists^2 = |c_i|^2 + |c_j|^2 - 2 c_i . c_j
        # sq_norms = np.sum(c**2, axis=1)
        # dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (c @ c.T)
        # dists = np.sqrt(np.maximum(dists_sq, 0))
        
        sq_norms = np.sum(c**2, axis=1)
        dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (c @ c.T)
        # Clip negative values due to numerical error
        dists_sq = np.maximum(dists_sq, 0)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_wall, min_pair / 2.0)
    
    # Update best
    current_radius = fast_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    # Local Optimization
    # We'll do a simple hill climbing with a fixed step size, but smarter.
    # Maybe random steps?
    
    # Let's try:
    # 1. Perturb randomly.
    # 2. Local polish.
    
    # Given time constraints, let's do a limited number of random restarts.
    
    rng = np.random.default_rng(42)
    
    for restart in range(10): # Try up to 10 restarts
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_centers = best_centers.copy()
        else:
            # Perturb best centers
            current_centers = best_centers + rng.normal(0, 0.05, size=(N, 3))
            # Clip to cube
            current_centers = np.clip(current_centers, -0.49, 0.49)
            
        current_radius = fast_radius(current_centers)
        
        # Local polish
        step = 0.01
        for it in range(50): # Limit iterations per restart
            if time.time() > deadline:
                break
                
            improved = False
            # Try moving each point
            # To speed up, we can sample a subset of points or directions?
            # Or just iterate through all.
            # 100 points * 6 dirs = 600 evaluations.
            # Each eval is O(N^2) = 10k ops.
            # 600 * 10k = 6M ops.
            # 50 iters * 6M = 300M ops.
            # Should be fast in numpy if vectorized? No, loop is in Python.
            # 600 Python loops per iter. 50 iters = 30k loops.
            # Each loop does matrix mult (100x100) and min.
            # This might take ~1ms per loop? 30ms per iter? 1.5s per restart.
            # 10 restarts = 15s. Acceptable.
            
            for i in range(N):
                for d in range(3):
                    for sign in [-1, 1]:
                        new_c = current_centers.copy()
                        new_c[i, d] += sign * step
                        
                        if np.any(np.abs(new_c) > 0.5):
                            continue
                            
                        new_r = fast_radius(new_c)
                        if new_r > current_radius + 1e-8:
                            current_centers = new_c
                            current_radius = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                # Try smaller step?
                step *= 0.5
                if step < 1e-5:
                    break
                    
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current_centers.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
