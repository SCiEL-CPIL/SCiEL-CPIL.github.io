"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Distance between all pairs
    # Use broadcasting for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # centers are in [-0.5, 0.5]^3
    # distance to nearest wall = 0.5 - max(|x|, |y|, |z|) for each center? No.
    # distance to wall k for center i is 0.5 - |c_ik|
    # min_i min_k (0.5 - |c_ik|)
    abs_centers = np.abs(centers)
    min_wall_dist = np.min(0.5 - abs_centers)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _init_grid(n):
    """Initialize with a simple grid, slightly perturbed."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Generate all grid points
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take first n
    idx = np.random.choice(len(grid), n, replace=False)
    centers = grid[idx].copy()
    # Add small noise
    centers += np.random.normal(0, 0.01, centers.shape)
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _local_refine(centers, max_iter=100, tol=1e-6):
    """Local refinement using gradient-free optimization (Nelder-Mead is too slow for 300 dims).
    Use a simple coordinate descent or simulated annealing step?
    Let's use a simple iterative repulsion method."""
    # Repulsion method: move each center away from others and walls
    for it in range(max_iter):
        moved = False
        for i in range(N):
            # Compute forces
            force = np.zeros(3)
            # Repulsion from other spheres
            for j in range(N):
                if i == j:
                    continue
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    d = np.random.normal(0, 1e-3, 3)
                    dist = np.linalg.norm(d)
                # We want dist >= 2*r. Currently r is determined by min dist.
                # Approximate force: push apart if close.
                # Let's use a potential energy approach.
                # But simple gradient descent on the objective is tricky.
                # Let's just do a local search: try to increase min distance.
                pass
            
            # This brute force local refinement is too slow and weak.
            # Let's switch strategy: use scipy.optimize.minimize with BFGS or L-BFGS-B on a smooth surrogate.
            break
        
        # Actually, let's implement a proper local optimization.
        # Objective: maximize r.
        # Equivalent to minimizing -r.
        # r = min( min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|) )
        # This is non-differentiable.
        # Use a smooth approximation? Or just use Nelder-Mead on subsets?
        # Given 150s limit, we can do a few iterations of a simple heuristic.
        
        # Heuristic: For each sphere, compute its "bottleneck" (closest neighbor or wall).
        # Move it slightly away from the bottleneck.
        
        # Calculate current radius
        r_current = _compute_radius(centers)
        
        # For each point, find closest constraint
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        # Find global minimum pair distance
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(min_wall_per_point)
        
        if min_pair_dist / 2.0 < min_wall_dist:
            # Bottleneck is a pair
            # Find all pairs with distance close to min
            # Move them apart
            # This is complex. Let's just do a random perturbation of the most constrained points.
            pass
        else:
            # Bottleneck is a wall
            pass
        
        # Simpler approach: Random local search (Hill Climbing)
        # Pick a random point, perturb it randomly, accept if radius increases.
        # This is slow in Python loops.
        
        # Let's use a vectorized approach for the objective evaluation in the loop.
        # But even that might be slow for 300 dims with many iterations.
        
        # Alternative: Use scipy.optimize.minimize on the whole 300D vector with a smooth objective.
        # Objective: sum of soft-min of (dist_ij - 2r) and (wall_dist_i - r).
        # Actually, maximizing the minimum is equivalent to minimizing the maximum violation.
        # Let's minimize: max( 2r - min_pair_dist, r - min_wall_dist )
        # This is still non-smooth.
        
        # Let's stick to a simple, fast heuristic:
        # 1. Compute r.
        # 2. Identify the "critical" points (those involved in the min distance or min wall distance).
        # 3. Move them slightly in a random direction that increases their distance from their bottleneck.
        
        # To keep it fast, we only do this for a few iterations.
        
        if it % 10 == 0:
            # Re-evaluate r to avoid drift
            pass
            
        # This local refinement is getting complicated and might be slow.
        # Let's rely on a better initialization and a global search like PSO or Differential Evolution if time permits,
        # or just a very good grid + small local jitter.
        
        # Given the constraints, let's implement a simple "Repulsion" step.
        # For each point i:
        #   Compute vector sum of (c_i - c_j) / |c_i - c_j|^2 for all j != i (Coulomb repulsion)
        #   Add vector pushing away from nearest wall.
        #   Move c_i by step_size * normalized_force.
        #   Project back to cube.
        
        step_size = 0.01
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (O(N^2))
        # This is 100^2 = 10000 ops per step. 100 steps = 1e6 ops. Fast enough.
        for i in range(N):
            # Repulsion from others
            diff = centers[i] - centers
            dists = np.linalg.norm(diff, axis=1)
            dists[i] = np.inf # avoid self
            # Inverse square law
            inv_dists_sq = 1.0 / (dists ** 2)
            inv_dists_sq[i] = 0
            force_rep = np.sum(diff * inv_dists_sq[:, np.newaxis], axis=0)
            
            # Wall repulsion
            force_wall = np.zeros(3)
            for k in range(3):
                if centers[i, k] > 0:
                    force_wall[k] = 1.0 / (0.5 - centers[i, k]) ** 2
                else:
                    force_wall[k] = -1.0 / (0.5 + centers[i, k]) ** 2
            
            forces[i] = force_rep + force_wall
        
        # Normalize forces?
        # forces = forces / (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-8)
        
        # Update
        new_centers = centers + step_size * forces
        
        # Clip to valid range (slightly inside to avoid boundary issues)
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        # Check if improvement
        new_r = _compute_radius(new_centers)
        if new_r > r_current:
            centers = new_centers
        else:
            # Try smaller step or random kick
            # For now, just stop if no improvement? Or keep trying with smaller step?
            # Let's try a smaller step.
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # 1. Initialization
    centers = _init_grid(N)
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # 2. Local Refinement
    if time.time() < deadline - 10:
        centers = _local_refine(centers, max_iter=200)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # 3. Global Search / Restart
    # Try a few restarts with different initializations if time permits.
    # Or a simple PSO?
    # Given 150s limit, and local refine is fast, we can do a few restarts.
    
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        # New random initialization
        # Mix of grid and random
        if restarts % 2 == 0:
            centers = _init_grid(N)
        else:
            # Random points in cube
            centers = np.random.uniform(-0.49, 0.49, (N, 3))
            
        # Local refine
        if time.time() < deadline - 10:
            centers = _local_refine(centers, max_iter=50) # Fewer iterations for restarts
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        if restarts > 10: # Cap restarts
            break

    return best_centers
# EVOLVE-BLOCK-END
