"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Physics-based Simulated Annealing to find a good packing for a 
target radius, followed by gradient-based polishing to maximize the radius.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    m = np.min(all_vals)
    
    # Weights for pairs
    pair_weights = np.exp(-pair_vals / t)
    np.fill_diagonal(pair_weights, 0.0)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    grad_pair_contrib = -0.5 * pair_weights[:, :, np.newaxis] * unit_vecs
    grad_c += np.sum(grad_pair_contrib, axis=1)
    
    # Wall contributions
    signs = np.sign(c)
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, max_iters=5000):
    """
    Perform repulsion-based simulated annealing.
    Minimizes a sum of inverse-square distances to encourage separation.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.05
    decay = 0.999
    
    # Initial energy calculation
    def _energy(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1e10)
        # Use a soft repulsion to avoid singularities and allow large moves
        # 1/d^2 is standard, but we add a small epsilon to avoid division by zero
        repulsion = np.sum(1.0 / (dists_sq + 1e-6))
        
        # Wall repulsion: push points away from walls
        # If a point is close to a wall, add penalty
        wall_dists = 0.5 - np.abs(c)
        # Penalize if wall dist is small. 
        # A smooth penalty: 1/wall_dist^2 is too strong near 0. 
        # Let's use a smooth potential that grows as we approach the wall.
        # Actually, standard SA for packing often just uses pair repulsion and clips to box.
        # But clipping can cause "sticking". Let's add a soft wall penalty.
        wall_penalty = np.sum(1.0 / (wall_dists + 1e-3)**2)
        
        return repulsion + wall_penalty

    E = _energy(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate move
        # Scale move size with temperature
        move_scale = 0.02 * temp
        move = np.random.normal(0.0, move_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.499, 0.499)
        
        E_new = _energy(c_new)
        
        # Metropolis acceptance
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / temp):
            c = c_new
            E = E_new
            
            # Check improvement
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                
        temp *= decay
        
    return best_c, best_r

def _lloyd_step(c):
    """
    One step of Lloyd's algorithm: move each point to the centroid of its Voronoi cell.
    Since scipy.spatial.Voronoi doesn't handle boundaries well for centroids directly,
    we approximate by using nearest neighbor regions or just use a simple gradient step
    towards the centroid of the "empty space".
    
    A more robust simple implementation for sphere packing:
    For each point, find its Voronoi region. The centroid of the region is the new position.
    We can approximate this by sampling or using the fact that the gradient of the sum of
    squared distances to the nearest neighbor is related to the Voronoi cell.
    
    Actually, a simpler and effective local search often used is "Voronoi relaxation" where
    we move points towards the center of their Voronoi cell.
    We can compute the Voronoi diagram and integrate over the cells.
    """
    from scipy.spatial import Voronoi, QhullError
    
    try:
        vor = Voronoi(c)
    except QhullError:
        return c
        
    new_c = c.copy()
    for i in range(N):
        # Get region vertices
        region = vor.regions[vor.point_region[i]]
        if len(region) == 0:
            continue
            
        verts = vor.vertices[region]
        
        # Check if region is bounded
        # If any vertex index is -1, the region is unbounded.
        # In our case, the cube is bounded, so Voronoi regions touching the boundary
        # are "cut off" by the cube. Standard Voronoi gives infinite regions.
        # We need to intersect with the cube.
        # This is complex. 
        # Alternative: Use a simple gradient step based on repulsion.
        # Or, use the fact that for a point in a bounded box, we can just clip the centroid.
        # But the centroid of an unbounded region is not well-defined.
        
        # Let's stick to a simpler local search if Voronoi is too complex to implement correctly with boundaries.
        # However, Lloyd's is very powerful. 
        # Let's try a different local search: "Centroid of nearest neighbors" or just Gradient Descent on Min Distance.
        
        pass
    
    return new_c

def _local_search_lloyd(c_init, deadline, max_iters=100):
    """
    Local search using gradient descent on the minimum distance (soft-min).
    This is effectively what _polish does, but we can make it more aggressive.
    """
    # We will rely on _polish for the main local optimization.
    # This function is a placeholder to maintain structure if we want to add specific Lloyd logic later.
    return c_init, _compute_radius(c_init)

def _polish(c_init, deadline, t_start=0.05, t_end=0.001, iters_per_temp=100):
    """
    Polish the configuration using gradient descent on the softmin radius with analytical gradients.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 10)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
            
        x0 = c.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        # Find the binding constraint
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        # If wall is binding, move the closest point away from the wall
        # If pair is binding, move the closest pair apart
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            # Direction away from wall
            sign = 1.0 if c[i, k] > 0 else -1.0
            # Try moving
            for step in (0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            # Pair binding
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. Random
    for i in range(3):
        seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
        
    # 2. Grid-like
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # We need 50, grid 4x4x4=64. Pick 50.
    # Randomly select 50 from grid
    idx = np.random.choice(len(grid), N, replace=False)
    seeds.append(grid[idx])
    
    # 3. FCC-like
    # Just a random perturbation of a structured start
    seeds.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    time_per_seed = (deadline - start_time) / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        # Stage 1: Repulsion Annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, max_iters=2000)
        
        # Stage 2: Polish
        c_polish, r_polish = _polish(c_anneal, seed_deadline, t_start=0.1, t_end=0.01)
        
        # Stage 3: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 2.0:
            break
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 5.0:
        final_deadline = deadline - 2.0
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.05, t_end=0.005)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 2.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
