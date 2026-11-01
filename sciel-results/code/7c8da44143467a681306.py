import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers in [-0.5, 0.5]^3.
    Returns the radius and the indices of the active constraints (closest pair and closest wall).
    """
    # Wall distance
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances using cKDTree for efficiency
    # We need the nearest neighbor for each point.
    # cKDTree.query returns distances to k nearest neighbors.
    # k=2 gives self (0) and nearest neighbor.
    try:
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        # Fallback if tree fails (e.g., duplicate points)
        min_pair_dist = 0.0

    r = min(min_wall_dist, min_pair_dist / 2.0)
    return r

def _gradient_step(centers, step_size, rng):
    """
    Perform a single gradient ascent step on the minimum radius.
    1. Identify the active constraints:
       - The pair of points with the minimum distance.
       - The point with the minimum wall distance.
    2. Compute a direction that increases these minimums.
    3. Move in that direction.
    """
    c = centers.copy()
    
    # 1. Identify constraints
    # Wall constraints
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    # Pair constraints
    try:
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        min_pair_dist = 0.0
        dists = None

    current_r = min(min_wall_dist, min_pair_dist / 2.0)
    
    # Gradient direction (sum of forces from active constraints)
    grad = np.zeros_like(c)
    
    # Force from wall constraints: Push points away from the closest wall
    # We look for points that are very close to the minimum wall distance
    # Threshold to include "near-active" constraints for smoother gradient
    wall_threshold = min_wall_dist + 0.05 
    close_to_wall = wall_dists < wall_threshold
    if np.any(close_to_wall):
        # For each point close to a wall, push it towards the center
        # grad[k] = -sign(c[k]) if c[k] is close to +/- 0.5
        for axis in range(3):
            mask_axis = close_to_wall[:, axis]
            if np.any(mask_axis):
                # Direction is towards 0
                grad[mask_axis, axis] -= np.sign(c[mask_axis, axis])
                # If exactly 0, push randomly or skip (sign is 0)
                # But sign(0) is 0, so it does nothing. That's fine.
                
    # Force from pair constraints: Push apart the closest pairs
    if dists is not None:
        # Find all pairs that are close to the minimum distance
        # We use the distances array from query (k=2)
        # dists[i, 1] is distance to nearest neighbor of i
        # We want to push i and its neighbor apart.
        # To be efficient, we can just take the global minimum pair and a few others.
        
        # Let's identify the specific pairs involved in the minimum distance
        # dists[:, 1] contains the distance to the nearest neighbor for each point.
        # The global min is min_pair_dist.
        # We want to push points i and j apart if dist(i,j) is small.
        
        # A simple approximation: For each point i, look at its nearest neighbor j.
        # If dist(i,j) is small, push i away from j.
        # Note: This might double count if j's nearest is i, but that's fine for gradient direction.
        
        # To get the indices of nearest neighbors, we need to modify the query or use a different approach.
        # cKDTree.query returns distances and indices if we ask for it? 
        # Actually, standard query returns (dist, idx) if we pass `k` and the tree is built.
        # Let's re-query to get indices.
        
        # Optimization: Only do this if pair distance is the bottleneck or close to it
        if min_pair_dist / 2.0 <= min_wall_dist + 0.05:
            # Re-query to get indices
            # We already have dists. We need indices.
            # scipy.spatial.cKDTree.query returns (dist, idx) by default? 
            # Yes, if k>1, it returns (dist, idx).
            # Wait, in the previous step I only unpacked dists. I should have kept idx.
            # Let's re-query here for clarity and correctness, it's fast for N=100.
            _, idx = tree.query(c, k=2)
            
            # idx[:, 1] is the index of the nearest neighbor for each point i
            for i in range(N):
                j = idx[i, 1]
                d = dists[i, 1]
                if d < min_pair_dist + 0.05: # Near-active pairs
                    diff = c[i] - c[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        # Push i away from j
                        grad[i] += diff / norm
                        # We don't necessarily need to push j here immediately because 
                        # when we process j, we might push j away from i.
                        # But to be symmetric and stronger, we can add to both.
                        # However, since we are summing gradients for all points, 
                        # just adding to i is a valid stochastic gradient step.
                        # Let's add to both for stronger effect.
                        grad[j] -= diff / norm

    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-10:
        # No clear gradient, add random noise
        grad = rng.normal(0, 1, c.shape)
        grad_norm = np.linalg.norm(grad)
    
    grad /= grad_norm
    
    # Move
    new_c = c + step_size * grad
    new_c = np.clip(new_c, -0.4999, 0.4999)
    
    return new_c

def _local_search(centers, deadline, rng):
    """
    Local search with gradient steps and adaptive step size.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Gradient step
        c_new = _gradient_step(c, step_size, rng)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r + 1e-10:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            no_improve_count = 0
            # Increase step size slightly if improving
            step_size = min(0.1, step_size * 1.05)
        else:
            no_improve_count += 1
            # Decrease step size
            step_size *= 0.95
            
            # If stuck, try a larger random jump to escape local optimum
            if no_improve_count > max_no_improve:
                # Random restart of a subset of particles
                kick_indices = rng.choice(N, size=10, replace=False)
                c[kick_indices] += rng.normal(0, 0.1, (10, 3))
                c = np.clip(c, -0.4999, 0.4999)
                
                r_curr = _compute_radius(c)
                if r_curr > best_r:
                    best_c = c.copy()
                    best_r = r_curr
                
                no_improve_count = 0
                step_size = 0.05 # Reset step size

    return best_c, best_r

def _init_grid():
    """
    Simple grid initialization.
    """
    # 5x5x4 grid = 100 points
    nx, ny, nz = 5, 5, 4
    # Scale to fit in [-0.4, 0.4] to leave room for optimization
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    init = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    return init

def _init_fcc():
    """
    FCC-like initialization.
    """
    points = []
    # Generate points in a 4x4x4 supercell of FCC
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # We will generate a grid of these
    for i in range(4):
        for j in range(4):
            for k in range(4):
                base = np.array([i*0.25, j*0.25, k*0.25])
                points.append(base)
                points.append(base + np.array([0, 0.125, 0.125]))
                points.append(base + np.array([0.125, 0, 0.125]))
                points.append(base + np.array([0.125, 0.125, 0]))
                
    points = np.array(points)
    # Center and scale
    points -= np.mean(points, axis=0)
    # Scale to fit in [-0.45, 0.45]
    max_coord = np.max(np.abs(points))
    if max_coord > 0.45:
        points *= 0.45 / max_coord
        
    # Shuffle and take first N
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(points))[:N]
    return points[idx]

def construct():
    start_time = time.time()
    # Safety margin: 145s is well within 150s.
    deadline = start_time + 145.0 
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple initializations
    inits = [
        _init_grid(),
        _init_fcc(),
    ]
    
    # Add perturbed versions of grid
    for i in range(2):
        perturbed = inits[0] + rng.normal(0, 0.05, inits[0].shape)
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)
        
    # Sort by initial radius to prioritize good starts
    init_radii = [_compute_radius(i) for i in inits]
    sorted_idx = np.argsort(init_radii)[::-1]
    inits = [inits[i] for i in sorted_idx]
    
    # Allocate time to each initialization
    # We have 4 inits. Give more time to the best ones.
    # Simple allocation: 40%, 30%, 20%, 10%
    weights = [0.4, 0.3, 0.2, 0.1]
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        # Calculate time slice
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        # Approximate slice based on remaining time and weight
        # This is a heuristic.
        slice_time = remaining * weights[i]
        slice_deadline = time.time() + slice_time
        
        # Ensure we don't overshoot the global deadline
        slice_deadline = min(slice_deadline, deadline - 1.0)
        
        c_opt, r_opt = _local_search(init, slice_deadline, rng)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
            
    # Final polish on the best found solution
    if best_centers is not None and time.time() < deadline - 2.0:
        # Use remaining time for final refinement
        c_final, r_final = _local_search(best_centers, deadline - 0.5, rng)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final.copy()
            
    if best_centers is None:
        # Fallback
        best_centers = inits[0]
        
    return best_centers

# EVOLVE-BLOCK-END
