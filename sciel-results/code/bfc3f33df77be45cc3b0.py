"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_grid(n):
    """Generate a jittered square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    # Use a grid slightly larger than needed to ensure we can pick n points
    # and have some room for jitter
    grid_size = m + 2 
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Randomly select n points from the grid and jitter them
    idxs = np.random.choice(len(grid), n, replace=False)
    points = grid[idxs].copy()
    
    # Jitter to break symmetry
    jitter = 0.01 / grid_size
    points += np.random.normal(0, jitter, (n, 2))
    
    # Ensure they are within bounds
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _force_directed_optimize(centers, deadline):
    """
    Optimize using force-directed dynamics (Molecular Dynamics).
    Uses a soft repulsive potential to maximize the minimum distance.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _score(centers)
    
    # Parameters
    # We want to maximize the minimum distance.
    # We can simulate this by applying forces that push points apart.
    # A common potential is V(r) = 1/r^2 or exp(-r).
    # Let's use a simple inverse square repulsion for close pairs.
    # However, to be efficient, we only need to consider neighbors.
    
    # Initial velocity is zero
    velocities = np.zeros((N, 2))
    
    # Temperature / Damping
    # We will use a cooling schedule or just fixed damping.
    # Let's try a simple Verlet-like integration with damping.
    
    dt = 0.01
    damping = 0.9
    
    # Force strength
    # We want forces to be significant when distance is small.
    # Let's define force F = k / d^2 * dir
    k = 0.01
    
    # Boundary force: strong repulsion from walls
    # F_wall = k_wall * (0.5 - |coord|) if near wall, else 0
    # Actually, simpler: if coord > 0.5 - margin, push back.
    k_wall = 10.0
    margin = 0.05
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        if time.time() > deadline:
            break
        
        # 1. Compute forces
        forces = np.zeros((N, 2))
        
        # Pairwise forces
        # For efficiency, we could use KDTree, but for N=200, O(N^2) vectorized is okay 
        # if we do it in chunks or if the constant is low.
        # However, 50000 steps * O(N^2) might be slow in pure Python loops, but vectorized NumPy is fast.
        # Let's try vectorized pairwise force calculation.
        
        # Calculate distances and differences
        # diff[i,j] = centers[i] - centers[j]
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        # Force magnitude: k / d^2
        # We only want repulsion, so force is always positive magnitude pushing apart.
        # F_ij = k / d^2 * (c_i - c_j) / d = k / d^3 * (c_i - c_j)
        # But we want to maximize min distance. A pure repulsion will push them until they hit walls or each other.
        # The equilibrium of pure repulsion in a box is not necessarily the max-min distance packing.
        # It tends to spread them out uniformly.
        # To bias towards max-min, we can use a "soft" constraint or just rely on the repulsion.
        # Let's stick to repulsion.
        
        inv_d3 = 1.0 / (dists_safe**3)
        # Zero out self-interaction
        np.fill_diagonal(inv_d3, 0.0)
        
        # Force on i due to j is k * inv_d3[i,j] * diffs[i,j]
        # Sum over j
        # forces[i] = sum_j k * inv_d3[i,j] * diffs[i,j]
        forces = k * np.sum(inv_d3[:, :, np.newaxis] * diffs, axis=1)
        
        # 2. Boundary forces
        # For each point, if it's close to a wall, add force pushing it back.
        for i in range(N):
            for dim in range(2):
                coord = centers[i, dim]
                if coord > 0.5 - margin:
                    forces[i, dim] -= k_wall * (coord - (0.5 - margin))
                elif coord < -0.5 + margin:
                    forces[i, dim] += k_wall * ((-0.5 + margin) - coord)
        
        # 3. Update velocities and positions
        velocities = velocities * damping + forces * dt
        centers += velocities * dt
        
        # 4. Clamp to bounds (hard constraint for stability)
        centers = np.clip(centers, -0.5, 0.5)
        
        # 5. Check score
        # To save time, check score every 10 steps
        if steps % 10 == 0:
            current_r = _score(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        steps += 1
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Initialize
    centers = _initial_grid(N)
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # 2. Optimization
    # Run force-directed optimization
    if time.time() < deadline - 5:
        centers, score = _force_directed_optimize(centers, deadline - 5)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # 3. Optional: Fine-tune with local search on the result?
    # The force-directed method might get stuck in local minima.
    # But given the budget, let's just return the best found.
    
    return best_centers
# EVOLVE-BLOCK-END
