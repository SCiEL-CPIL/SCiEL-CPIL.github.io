"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
import sys

def _compute_radius(centres):
    """
    Compute the maximum valid radius for a set of centres.
    Returns 0 if invalid (out of bounds or NaN).
    """
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds
    if np.any(centres < -0.5 - 1e-10) or np.any(centres > 0.5 + 1e-10):
        return 0.0
    
    c = np.clip(centres, -0.5, 0.5)
    
    # Wall constraint
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    if min_wall < -1e-10:
        return 0.0
        
    # Pairwise constraint using KD-tree for speed
    tree = cKDTree(c)
    # Query k=2 (self + 1 neighbor)
    # If a point has no neighbor within a certain range, we might need to expand, 
    # but for packing, nearest neighbor is the bottleneck.
    # Actually, for 100 points, cdist is O(N^2) = 10k ops, which is fast enough in numpy.
    # But KDTree query is O(N log N). Let's use cdist for simplicity and robustness, 
    # unless N is huge. N=100 is small.
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    r = min(min_wall, min_pair)
    return max(0.0, r)

def _force_directed_relax(centres, steps=2000, base_lr=0.01, cooling=0.999):
    """
    Relax the packing using force-directed dynamics.
    Forces:
    1. Repulsion between spheres if overlapping (or too close).
    2. Repulsion from walls if too close.
    """
    c = centres.copy().astype(float)
    lr = base_lr
    
    for step in range(steps):
        # Compute forces
        forces = np.zeros_like(c)
        
        # 1. Wall forces
        # Distance to nearest wall
        # x: dist to -0.5 is c[:,0] + 0.5, to 0.5 is 0.5 - c[:,0]
        dist_to_walls = np.stack([
            c + 0.5,
            0.5 - c
        ], axis=2) # shape (N, 3, 2)
        
        min_wall_dist = np.min(dist_to_walls, axis=2) # shape (N, 3)
        
        # If a sphere is very close to a wall, push it away
        # We want to maximize the minimum distance.
        # Force is proportional to how much it violates a "soft" target radius?
        # No, we just want to push away from walls to allow larger radius.
        # Let's use a potential that penalizes being close to walls.
        # Actually, the objective is min(wall_dist, pair_dist/2).
        # A good potential is sum of (target_r - dist)^2 for all constraints.
        # But we don't know target_r.
        # Alternative: Simple repulsion.
        
        # Let's use a simpler heuristic:
        # Push spheres away from walls if they are within 'margin'
        margin = 0.05
        for i in range(3):
            # Left/Bottom wall
            close_left = c[:, i] < -0.5 + margin
            forces[close_left, i] += (margin + 0.5 - c[close_left, i]) * 0.5
            # Right/Top wall
            close_right = c[:, i] > 0.5 - margin
            forces[close_right, i] -= (c[close_right, i] - (0.5 - margin)) * 0.5

        # 2. Pairwise repulsion
        # Use KDTree to find neighbors
        tree = cKDTree(c)
        # We need to find pairs closer than some threshold.
        # Since we are maximizing min dist, we care about the closest pairs.
        # Let's query all pairs within a reasonable distance, e.g., 0.2
        # For N=100, cdist is fast enough (100x100 = 10k distances).
        dists = cdist(c, c)
        
        # Repulsive force: F = k * (1/d^2) * dir, if d < threshold
        # We want to push them apart if they are too close.
        # "Too close" is relative to the current best radius?
        # Let's just push all pairs that are closer than 0.15.
        threshold = 0.15
        mask = (dists < threshold) & (dists > 1e-10)
        
        # Vectorized force calculation
        # F_ij = (c_i - c_j) / d^3  (inverse square law magnitude, unit vector direction)
        # Actually, gradient of 1/d is -1/d^2 * (c_i-c_j)/d.
        # Let's use a soft potential: V = 1/d^2 if d < threshold, else 0.
        # Force = -grad V = 2/d^3 * (c_i - c_j)
        
        # To avoid division by zero and singularities, add epsilon
        dists_safe = np.where(dists < 1e-10, 1.0, dists)
        
        # Only apply force if within threshold
        # F_ij = 2 * (c_i - c_j) / d^3
        # We can compute this matrix-wise
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # This is 100x100x3 = 30k floats. Fine.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        
        # d^3
        d3 = dists_safe[:, :, np.newaxis] ** 3
        
        # Force matrix
        force_mat = 2.0 * diff / d3
        
        # Zero out forces where dist >= threshold
        force_mat[~mask] = 0.0
        
        # Sum forces for each particle
        # forces[i] = sum_j F_ij
        forces += np.sum(force_mat, axis=1)
        
        # Update positions
        c += lr * forces
        
        # Clip to bounds (hard constraint)
        c = np.clip(c, -0.5, 0.5)
        
        # Annealing
        lr *= cooling
        
        # Optional: Check progress and break early if stuck?
        # For now, run all steps.
        
    return c

def _local_refine_fast(centres, steps=500):
    """
    Fast local refinement using gradient ascent on the min-distance metric.
    This is a simplified version of force-directed, specifically tuned to increase the minimum distance.
    """
    c = centres.copy()
    r = _compute_radius(c)
    
    # Identify the "bottleneck" constraints
    # 1. Wall bottlenecks
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # 2. Pair bottlenecks
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    current_r = min(min_wall, min_pair_dist / 2.0)
    
    # We will try to increase the minimum distance.
    # Strategy: Move the particles involved in the minimum distance apart.
    # Also move particles close to walls away from walls.
    
    for _ in range(steps):
        # Recalculate bottlenecks
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        current_r = min(min_wall, min_pair_dist / 2.0)
        
        # If wall is the bottleneck
        if min_wall < min_pair_dist / 2.0:
            i, k = min_wall_idx
            # Move particle i away from wall
            # Determine which wall
            if c[i, k] < 0:
                c[i, k] += 0.001
            else:
                c[i, k] -= 0.001
        else:
            # Pair is the bottleneck
            i, j = min_pair_idx
            # Move i and j apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec_norm = vec / norm
                c[i] += vec_norm * 0.0005
                c[j] -= vec_norm * 0.0005
        
        # Clip
        c = np.clip(c, -0.5, 0.5)
        
        # Check if improved
        new_r = _compute_radius(c)
        if new_r > current_r + 1e-12:
            continue
        else:
            # If not improved, try a smaller step or different direction?
            # For now, just continue. This is a simple hill climber.
            pass
            
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    np.random.seed(42)
    
    best_centres = None
    best_r = -1.0
    
    # Strategy 1: Grid start
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    # Relax the grid
    grid_relaxed = _force_directed_relax(grid, steps=1000, base_lr=0.005)
    r_grid = _compute_radius(grid_relaxed)
    if r_grid > best_r:
        best_r = r_grid
        best_centres = grid_relaxed.copy()
        
    # Local refine from grid
    grid_refined = _local_refine_fast(grid_relaxed, steps=200)
    r_grid_ref = _compute_radius(grid_refined)
    if r_grid_ref > best_r:
        best_r = r_grid_ref
        best_centres = grid_refined.copy()
        
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_centres
        
    # Strategy 2: Random restarts with Force-Directed Relaxation
    # This is the main improvement: FD relaxation explores the space better than greedy hill climbing.
    n_restarts = 0
    max_restarts = 50
    
    while time.time() - start_time < deadline - 10:
        n_restarts += 1
        
        # Generate random start
        # Use a slightly structured random start to help convergence
        # e.g. Random points in a smaller sub-cube, then expand?
        # Or just uniform random.
        c_rand = np.random.uniform(-0.4, 0.4, size=(N, 3))
        
        # Relax
        c_relaxed = _force_directed_relax(c_rand, steps=500, base_lr=0.01, cooling=0.999)
        
        # Check radius
        r_relaxed = _compute_radius(c_relaxed)
        
        # If it's a candidate for best, do local refinement
        if r_relaxed > best_r * 0.9: # Only refine if promising
             c_refined = _local_refine_fast(c_relaxed, steps=300)
             r_refined = _compute_radius(c_refined)
             if r_refined > best_r:
                 best_r = r_refined
                 best_centres = c_refined.copy()
                 
        # Also try refining the raw relaxed version if it's very good
        if r_relaxed > best_r:
             best_r = r_relaxed
             best_centres = c_relaxed.copy()
             
        # Periodically refine the current best to see if we can squeeze more
        if n_restarts % 5 == 0 and best_centres is not None:
             c_best_ref = _local_refine_fast(best_centres, steps=100)
             r_best_ref = _compute_radius(c_best_ref)
             if r_best_ref > best_r:
                 best_r = r_best_ref
                 best_centres = c_best_ref.copy()
                 
    # Final local refinement on the best solution found
    if best_centres is not None and time.time() - start_time < deadline - 5:
        c_final = _local_refine_fast(best_centres, steps=500)
        r_final = _compute_radius(c_final)
        if r_final > best_r:
            best_r = r_final
            best_centres = c_final.copy()
            
    return best_centres
# EVOLVE-BLOCK-END
