"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import minimize

N_CIRCLES = 200
HALF_SIDE = 0.5

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0
    
    # Distance to boundary
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0
    
    # Min pairwise distance
    tree = cKDTree(centers)
    # Use a generous radius for query_pairs to ensure we catch the minimum
    # Since we expect r ~ 0.036, 2*r ~ 0.073. 0.1 is safe.
    pairs = tree.query_pairs(r=0.1) 
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def _simple_grid():
    """Fallback simple grid."""
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]

def voronoi_relaxation(centers, max_iters=100, step_size=0.01, time_limit=60.0):
    """
    Iteratively relax centers towards Voronoi cell centroids.
    This tends to equalize local density and push circles apart.
    """
    centers = centers.copy()
    start_time = time.time()
    
    # Add boundary points to handle Voronoi cells touching the square edges correctly
    # We add points outside the square to "close" the cells for boundary circles
    # Actually, for inradius calculation, we need to consider the square boundary.
    # Standard Voronoi in scipy is for infinite plane.
    # To handle boundaries, we can add "dummy" points far away or use a different method.
    # A simpler robust method for radius estimation:
    # r_i = min( dist(c_i, c_j) for all j!=i ) / 2, AND dist(c_i, boundary)
    # But Voronoi centroid relaxation is powerful for spacing.
    
    # Let's use a simpler relaxation: 
    # 1. Compute "pressure" or force from neighbors.
    # 2. Move away from close neighbors.
    # This is basically Force-Directed Graph layout or Simple Annealing.
    
    # Let's try a specific spectral-inspired gradient step:
    # We want to maximize the minimum distance.
    # Gradient of min_dist is tricky.
    # Let's stick to a robust "Repulsion + Boundary Attraction" physics step.
    
    best_r = compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if time.time() > start_time + time_limit:
            break
            
        # 1. Find close pairs (potential collisions or tight constraints)
        tree = cKDTree(centers)
        # Look for pairs closer than 2 * best_r * 1.1
        query_r = max(2 * best_r * 1.1, 0.05)
        pairs = tree.query_pairs(r=query_r)
        
        forces = np.zeros_like(centers)
        
        # 2. Apply repulsive forces for close pairs
        for i, j in pairs:
            d_vec = centers[i] - centers[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-9:
                # Random kick if coincident
                forces[i] += np.random.uniform(-1, 1, 2)
                forces[j] -= np.random.uniform(-1, 1, 2)
                continue
            
            # Repulsion strength increases as distance decreases
            # Ideal distance is 2 * target_r. We don't know target_r exactly, 
            # but we can use current best_r as a proxy for "minimum allowed".
            # If d < 2 * best_r, push apart strongly.
            if d < 2 * best_r:
                # Strong repulsion
                strength = (2 * best_r - d) / d
                forces[i] += d_vec * strength
                forces[j] -= d_vec * strength
            else:
                # Weak attraction to neighbors to keep them close? 
                # No, we want to maximize global min dist. 
                # Just gentle repulsion to maintain spacing.
                strength = 0.1 / d
                forces[i] += d_vec * strength
                forces[j] -= d_vec * strength
        
        # 3. Boundary attraction/constraint
        # If a circle is too close to the wall (dist < best_r), push it in.
        dist_to_left = centers[:, 0] + HALF_SIDE
        dist_to_right = HALF_SIDE - centers[:, 0]
        dist_to_bottom = centers[:, 1] + HALF_SIDE
        dist_to_top = HALF_SIDE - centers[:, 1]
        
        # Push away from walls if violation
        forces[:, 0] += np.where(dist_to_left < best_r, 1.0, 0.0)
        forces[:, 0] -= np.where(dist_to_right < best_r, 1.0, 0.0)
        forces[:, 1] += np.where(dist_to_bottom < best_r, 1.0, 0.0)
        forces[:, 1] -= np.where(dist_to_top < best_r, 1.0, 0.0)
        
        # 4. Normalize forces to prevent huge jumps
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        forces = forces / norms
        
        # 5. Update positions
        # Adaptive step size: smaller if we improved recently, larger if stuck
        # Simple heuristic: step_size * 0.5 * (1 + noise)
        actual_step = step_size * 0.1 
        centers += forces * actual_step
        
        # 6. Project back to square (hard constraint)
        centers = np.clip(centers, -HALF_SIDE, HALF_SIDE)
        
        # 7. Evaluate
        current_r = compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            # If we improved, maybe increase step slightly? 
            # Or keep it stable.
            # Let's try to increase step size slowly if we keep improving
            step_size = min(step_size * 1.05, 0.05)
        else:
            # If no improvement, revert to best and decrease step or randomize
            centers = best_centers.copy()
            step_size *= 0.9
            # Add small random perturbation to escape local minima
            centers += np.random.normal(0, 0.001, size=centers.shape)
            centers = np.clip(centers, -HALF_SIDE, HALF_SIDE)
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s
    
    # 1. Initialize with a grid
    best_config = _simple_grid()
    best_r = compute_radius(best_config)
    
    # 2. Run Voronoi/Physics Relaxation
    # Use multiple restarts with different initializations if time permits
    inits = [
        _simple_grid(),
        _simple_grid() + np.random.uniform(-0.01, 0.01, size=(N_CIRCLES, 2))
    ]
    
    for init in inits:
        init = np.clip(init, -HALF_SIDE, HALF_SIDE)
        if time.time() > deadline - 20.0:
            break
            
        # Run relaxation
        # We allocate remaining time minus buffer
        remaining = deadline - 10.0 - time.time()
        if remaining <= 0:
            break
            
        new_config, new_r = voronoi_relaxation(init, max_iters=200, step_size=0.02, time_limit=remaining)
        
        if new_r > best_r:
            best_r = new_r
            best_config = new_config
    
    # Final check
    final_r = compute_radius(best_config)
    if final_r < best_r:
        # Should not happen if we track best, but safety
        pass
        
    return best_config
# EVOLVE-BLOCK-END
