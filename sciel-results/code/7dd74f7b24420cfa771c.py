"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius_full(c):
    """
    Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3.
    Returns r and the distance matrix for potential reuse.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # c is (N, 3)
    # diff is (N, N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_pair / 2.0, min_wall)
    return r, dists

def _initial_config(n):
    """
    Generate a good initial configuration.
    We use a 4x4x4 grid (64 points) and remove 14 points strategically 
    or just use a perturbed grid. 
    To maximize radius, points on the boundary are good.
    """
    # 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 64 points. We need 50.
    # Removing points from the corners might reduce the min distance if corners are close to edges?
    # Actually, in a 4x4x4 grid, the spacing is 1/3.
    # Distance between neighbors is 1/3.
    # Radius is 1/6 = 0.1666? No, wait.
    # If points are at -0.5, -0.166, 0.166, 0.5.
    # Dist between -0.5 and -0.166 is 0.333.
    # Radius 0.166.
    # But we have 50 points. If we just take the first 50, we break symmetry.
    # A better initialization for SA is a random one or a perturbed grid.
    # Let's stick to the grid but shuffle and take 50, then add noise.
    np.random.seed(42)
    idx = np.random.choice(len(grid), n, replace=False)
    c = grid[idx].copy()
    # Add small noise to break symmetry
    c += np.random.uniform(-0.01, 0.01, size=c.shape)
    c = np.clip(c, -0.499, 0.499)
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin for 150s limit
    
    # 1. Initial Configuration
    c = _initial_config(N)
    r, dists = _compute_radius_full(c)
    
    best_c = c.copy()
    best_r = r
    
    # Simulated Annealing Parameters
    # We want to maximize r.
    # Objective function: f(c) = r(c)
    # We accept move if delta_r > 0 or with probability exp(delta_r / T)
    # Since r is small (~0.13), delta_r will be very small.
    # We should scale the objective or use a different acceptance criterion.
    # Let's use the "energy" E = -r. We want to minimize E.
    # delta_E = -delta_r.
    # Accept if delta_E < 0 (improvement) or exp(-delta_E/T) > random.
    
    # Initial Temperature
    # Typical change in r is ~1e-4 to 1e-3.
    # Let's start T such that exp(-1e-3/T) is significant.
    T = 1e-3
    T_min = 1e-7
    cooling_rate = 0.999
    
    # Step size for moves
    # Move one point by a random vector.
    step_size = 0.05
    
    current_r = r
    current_dists = dists
    
    # Precompute indices for faster updates?
    # For N=50, updating one row/col of a 50x50 matrix is fast.
    
    n_iterations = 0
    no_improve_count = 0
    best_no_improve_T = T
    
    while time.time() < deadline:
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Generate random displacement
        # Gaussian step
        disp = np.random.normal(0, step_size, size=3)
        
        # Ensure the move stays within the cube
        # We can just clip, but clipping changes the effective step.
        # Better: if c_i + disp is out of bounds, reject or reflect?
        # Rejecting is simpler.
        new_c_i = c[i] + disp
        if np.any(new_c_i < -0.5) or np.any(new_c_i > 0.5):
            # Try a smaller step or just skip
            continue
            
        # Calculate new radius
        # Update distances for point i
        # dists[i, :] and dists[:, i]
        # diff = c[i] - c[j] for all j != i
        diff_i = c[i] - c
        diff_i[i] = 0 # Self distance 0
        new_dists_i = np.sqrt(np.sum(diff_i * diff_i, axis=1))
        
        # The new minimum pair distance is min(
        #   min(new_dists_i[j] for j != i), 
        #   min(dists[k, l] for k,l != i and k != l)
        # )
        # We don't have the "min excluding i" precomputed easily without a heap or re-computation.
        # However, N=50 is small. 
        # Option A: Recompute full min. (Fast in NumPy)
        # Option B: Maintain a heap of distances. (Complex)
        # Let's try Option A first, it's O(N^2) which is 2500 ops. 
        # 100k iterations * 2500 ops = 2.5e8 ops. Might be too slow in Python loop?
        # NumPy vectorization helps.
        
        # Let's just compute the new full radius if the local min looks promising?
        # Or just compute it.
        
        # Actually, we can compute the new global min pair distance more efficiently:
        # new_min_pair = min( min(new_dists_i[1:]), min(new_dists_i[:i]), 
        #                     min of dists matrix excluding row/col i )
        
        # Let's extract the sub-matrix excluding i
        # This is a bit expensive in Python to slice and min every time.
        
        # Alternative: Just recompute _compute_radius_full on a modified copy?
        # That copies the array. 
        # c_new = c.copy()
        # c_new[i] = new_c_i
        # new_r, _ = _compute_radius_full(c_new)
        # This is clean. Let's see if it's fast enough.
        # 50x3 array copy is tiny. 
        # 50x50 distance calc is the main cost.
        
        c_new = c.copy()
        c_new[i] = new_c_i
        new_r, _ = _compute_radius_full(c_new)
        
        delta_r = new_r - current_r
        
        # Acceptance criterion
        if delta_r > 0:
            # Improvement
            c = c_new
            current_r = new_r
            no_improve_count = 0
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        else:
            # Worse or same
            # Accept with probability exp(delta_r / T)
            # delta_r is negative.
            if np.random.rand() < np.exp(delta_r / T):
                c = c_new
                current_r = new_r
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cooling
        # Cool faster if we've been stuck? Or standard geometric?
        T *= cooling_rate
        
        # If T is too low, we might get stuck. 
        # Restart?
        if T < T_min:
            T = T_min
            
        n_iterations += 1
        
        # Periodic restart if stuck for a long time
        if no_improve_count > 5000:
            # Perturb the current solution significantly
            # Or reset to best_c with noise?
            # Let's try to "kick" the current solution
            kick_idx = np.random.randint(N)
            c[kick_idx] += np.random.normal(0, 0.1, size=3)
            c = np.clip(c, -0.499, 0.499)
            current_r, _ = _compute_radius_full(c)
            T = T_min * 10 # Reheat slightly
            no_improve_count = 0
            step_size *= 1.5 # Increase step size to escape
            step_size = min(step_size, 0.2)

    # Final check
    best_r, _ = _compute_radius_full(best_c)
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c

# EVOLVE-BLOCK-END
