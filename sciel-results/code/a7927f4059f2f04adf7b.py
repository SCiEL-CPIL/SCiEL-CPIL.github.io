"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    def compute_true_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def repulsive_energy(c):
        """Sum of inverse square distances. Lower is better (more spread out)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero, doesn't matter as we ignore diagonal
        # Use a softening parameter to avoid explosion if points get too close
        dists_sq_safe = dists_sq + 1e-6
        # We want to minimize sum of 1/d^2
        # Gradient of sum 1/d^2 w.r.t c_i:
        # d/dc_i (1/|c_i-c_j|^2) = -2 * (c_i - c_j) / |c_i-c_j|^4
        # So gradient of energy is sum_j -2 * (c_i - c_j) / d^4
        # We want to move AGAINST the gradient to minimize energy.
        # So direction = -grad = sum_j 2 * (c_i - c_j) / d^4
        # Wait, if c_i > c_j, (c_i - c_j) is positive. 1/d^2 increases as they get closer.
        # So if they are close, we want to push them apart.
        # Force on i from j is repulsive: direction (c_i - c_j).
        # Energy U = sum 1/d^2. Grad U_i = sum_j -2 (c_i - c_j) / d^4.
        # To minimize U, we move c_i -= lr * Grad U_i.
        # c_i += lr * sum_j 2 (c_i - c_j) / d^4.
        
        # Vectorized gradient calculation
        # diff is (N, N, 3)
        # dists_sq is (N, N)
        # We need weights w_ij = 2 / dists_sq_safe[i,j]^2
        # grad_i = sum_j w_ij * diff[i,j]
        
        weights = 2.0 / (dists_sq_safe**2)
        np.fill_diagonal(weights, 0.0)
        
        # grad = einsum('ij,ijk->ik', weights, diff)
        grad = np.einsum('ij,ijk->ik', weights, diff)
        
        # Energy value for tracking
        energy = np.sum(weights) # This is proportional to sum 1/d^2? No, weights are 2/d^4.
        # Let's just return the gradient. The energy value is less important than the direction.
        return grad

    def maximin_steepest_ascent(c, lr):
        """
        Move points to increase the minimum distance.
        Strategy:
        1. Find the minimum distance pair (i, j).
        2. Push i and j apart.
        3. For all other points, compute a "repulsion" from the nearest neighbors to prevent new collisions.
        """
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Find the global minimum distance
        min_dist = np.min(dists)
        
        # Find all pairs that are close to the minimum (within 1% or so)
        # This helps if there are multiple degenerate minima
        threshold = min_dist * 1.01
        close_mask = dists <= threshold
        close_mask = close_mask & ~np.eye(N, dtype=bool)
        
        # For each point i, sum the vectors pointing away from close neighbors j
        # Vector v_ij = (c_i - c_j) / |c_i - c_j|
        # We want to move i in direction sum_j v_ij * weight
        # Weight should be stronger for closer pairs.
        
        # Safe dists for division
        dists_safe = np.where(dists == 0, 1.0, dists)
        
        # Unit vectors
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        
        # Weights: 1 for close pairs, 0 otherwise? 
        # Or 1/d for all pairs?
        # Let's use 1/d for all pairs, but boost the close ones.
        # Actually, standard steepest ascent for max-min is tricky.
        # Let's use a force model: F_i = sum_j (c_i - c_j) / d_ij^3 * k
        # This is like a spring force.
        
        # Let's stick to the gradient of the smooth min from the previous approach but simpler:
        # We will just use the repulsive gradient but scaled by distance.
        # Actually, let's just use the repulsive gradient from the previous step, but with a steeper potential: 1/d^4
        # This effectively pushes the closest pairs apart most strongly.
        
        diff_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(diff_sq, 1.0)
        
        # Force = diff / d^4
        # This is equivalent to gradient of 1/d^2 * 2, which is what we had in repulsive_energy
        # But let's make it steeper: 1/d^6 (gradient of 1/d^4)
        # Grad of 1/d^4 = -4 d^-5 * (c_i - c_j)/d = -4 (c_i - c_j) / d^6
        # So move c_i += lr * 4 * sum_j (c_i - c_j) / d^6
        
        weights = 4.0 / (diff_sq**3 + 1e-6)
        np.fill_diagonal(weights, 0.0)
        
        grad = np.einsum('ij,ijk->ik', weights, diff)
        
        # Also handle walls: if a point is close to a wall, push it in.
        # Wall distance w_ik = 0.5 - |c_ik|
        # If w_ik < min_dist/2 (or some threshold), push it towards center.
        # Simple heuristic: add a force towards center proportional to proximity to wall.
        # Force_wall = -sign(c) * (1 - 2*|c|)^2 ?
        # Let's just clip and rely on the repulsion to handle most of it, 
        # but add a small bias towards center if very close to wall.
        
        wall_dist = 0.5 - np.abs(c)
        # If wall_dist is small, we want to move towards 0.
        # Direction towards 0 is -sign(c).
        # Magnitude: 1/wall_dist ?
        
        # Let's just use the repulsive gradient for now, it's robust.
        return grad

    def optimize_single(c_init, time_budget):
        c = c_init.copy()
        # Phase 1: Repulsion (1/d^2) to spread out
        # Phase 2: Steeper Repulsion (1/d^4) to resolve close contacts
        
        # Adaptive learning rate
        lr = 0.05
        
        best_c = c.copy()
        best_r = compute_true_radius(c)
        
        steps = 0
        while time.time() < start_time + time_budget:
            # Alternate between soft and hard repulsion
            if steps % 10 == 0:
                # Soft repulsion for global arrangement
                grad = repulsive_energy(c)
            else:
                # Hard repulsion for local fixes
                grad = maximin_steepest_ascent(c, lr)
            
            # Normalize gradient to prevent huge jumps
            norm = np.linalg.norm(grad)
            if norm > 0:
                grad = grad / norm
            
            c += lr * grad
            
            # Clip to bounds
            c = np.clip(c, -0.5, 0.5)
            
            # Evaluate
            if steps % 5 == 0:
                r = compute_true_radius(c)
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
            
            # Decay learning rate
            lr *= 0.995
            steps += 1
            
        return best_c

    # Generate multiple initial configurations
    
    # 1. Grid
    nx, ny, nz = 5, 5, 4
    x_coords = (np.arange(nx) + 0.5) / nx - 0.5
    y_coords = (np.arange(ny) + 0.5) / ny - 0.5
    z_coords = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    c_grid = grid[:N].copy()
    
    # 2. Fibonacci Sphere mapped to Cube
    # Generate points on sphere, then project/scale to cube?
    # Better: Generate points in cube using Fibonacci lattice in 3D?
    # Simple random with fixed seed
    np.random.seed(123)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    # 3. Jittered Grid
    np.random.seed(456)
    c_jitter = c_grid + np.random.normal(0, 0.02, size=(N, 3))
    c_jitter = np.clip(c_jitter, -0.45, 0.45)

    # Run optimization on each
    # Allocate time: 40s for each of 3 runs = 120s. Total 145s limit.
    time_per_run = 40.0
    
    candidates = [c_grid, c_rand, c_jitter]
    best_overall = None
    best_score_overall = 0.0
    
    for i, c_init in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        c_opt = optimize_single(c_init, time_per_run)
        r = compute_true_radius(c_opt)
        
        if r > best_score_overall:
            best_score_overall = r
            best_overall = c_opt
            
        # Check time
        if time.time() > deadline - 10:
            break

    # If we didn't get a result (shouldn't happen), return grid
    if best_overall is None:
        best_overall = c_grid
        
    return best_overall
# EVOLVE-BLOCK-END
