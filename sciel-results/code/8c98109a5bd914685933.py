"""Simulated annealing packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import random

N = 50

def construct():
    # Seed random for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initial positions: hexagonal lattice distorted slightly
    # Generate hexagonal lattice points
    points = []
    r_hex = 0.15  # approximate spacing
    sqrt3 = np.sqrt(3.0)
    
    # Generate hex grid
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * r_hex + (j % 2) * r_hex / 2.0
            y = j * r_hex * sqrt3 / 2.0
            if x*x + y*y <= 0.9:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, select N closest to center
    if len(points) > N:
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points if needed
        while len(points) < N:
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 0.8)
            points = np.vstack([points, [[r*np.cos(angle), r*np.sin(angle)]]])
    
    # Simulated annealing
    def energy(pos):
        """Compute energy: penalize overlaps and container violations."""
        pos = pos.reshape(-1, 2)
        # Container violations
        dists = np.sqrt(np.sum(pos**2, axis=1))
        container_violation = np.sum(np.maximum(0, dists - 1.0))
        
        # Circle-circle overlaps
        # Compute all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Overlap penalty: for each pair, if dist < 2*r, penalize
        # We'll use a fixed target radius for penalty calculation
        target_r = 0.125
        overlap = np.sum(np.maximum(0, 2*target_r - dist_mat))
        
        return container_violation * 10.0 + overlap
    
    def move(pos, step_size):
        """Random move."""
        new_pos = pos.copy().reshape(-1, 2)
        idx = np.random.randint(0, N)
        new_pos[idx] += np.random.normal(0, step_size, 2)
        return new_pos.reshape(-1)
    
    pos = points.reshape(-1).copy()
    T = 0.1
    T_min = 1e-6
    cooling = 0.9995
    max_iter = 50000
    step_size = 0.05
    
    best_pos = pos.copy()
    best_energy = energy(pos)
    
    for it in range(max_iter):
        new_pos = move(pos, step_size)
        e_old = energy(pos)
        e_new = energy(new_pos)
        
        delta_e = e_new - e_old
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            pos = new_pos
            if e_new < best_energy:
                best_energy = e_new
                best_pos = new_pos.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Adjust step size
        if it % 1000 == 0:
            step_size *= 0.99
    
    # Refine with local optimization
    pos = best_pos.copy()
    
    def objective(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        dists = np.sqrt(np.sum(pos**2, axis=1))
        container_violation = np.sum(np.maximum(0, dists - 1.0))
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        target_r = 0.125
        overlap = np.sum(np.maximum(0, 2*target_r - dist_mat))
        
        return container_violation * 10.0 + overlap
    
    result = minimize(objective, pos, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-10})
    
    if result.fun < best_energy:
        pos = result.x
    else:
        pos = best_pos
    
    pos = pos.reshape(-1, 2)
    
    # Compute the actual radius
    # Find minimum pairwise distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    
    # Find minimum distance to boundary
    dists = np.sqrt(np.sum(pos**2, axis=1))
    min_boundary_dist = np.min(1.0 - dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Scale positions to maximize radius
    # If we can scale, do so
    if r > 0:
        # Check if we can scale up
        # The limiting factor is either pair distance or boundary
        scale = 1.0
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Pair distance is limiting, can scale until boundary is also limiting
            # But scaling increases both, so we need to find optimal scale
            # Actually, scaling by s: pair_dist becomes s*pair_dist, boundary becomes 1 - s*|c|
            # We want to maximize min(s*pair_dist/2, min_i(1 - s*|c_i|))
            # This is more complex, so let's just try a few scales
            for s in [1.0, 1.05, 1.1, 1.15, 1.2]:
                scaled_pos = pos * s
                scaled_dists = np.sqrt(np.sum(scaled_pos**2, axis=1))
                if np.max(scaled_dists) > 1.0:
                    continue
                scaled_diff = scaled_pos[:, np.newaxis, :] - scaled_pos[np.newaxis, :, :]
                scaled_dist_mat = np.sqrt(np.sum(scaled_diff**2, axis=2))
                np.fill_diagonal(scaled_dist_mat, np.inf)
                scaled_min_pair = np.min(scaled_dist_mat)
                scaled_r = min(scaled_min_pair / 2.0, np.min(1.0 - scaled_dists))
                if scaled_r > r:
                    r = scaled_r
                    pos = scaled_pos
        else:
            # Boundary is limiting
            for s in [1.0, 0.95, 0.9, 0.85]:
                scaled_pos = pos * s
                scaled_dists = np.sqrt(np.sum(scaled_pos**2, axis=1))
                scaled_diff = scaled_pos[:, np.newaxis, :] - scaled_pos[np.newaxis, :, :]
                scaled_dist_mat = np.sqrt(np.sum(scaled_diff**2, axis=2))
                np.fill_diagonal(scaled_dist_mat, np.inf)
                scaled_min_pair = np.min(scaled_dist_mat)
                scaled_r = min(scaled_min_pair / 2.0, np.min(1.0 - scaled_dists))
                if scaled_r > r:
                    r = scaled_r
                    pos = scaled_pos
    
    # Final validation
    dists = np.sqrt(np.sum(pos**2, axis=1))
    if np.any(dists > 1.0):
        # Project back inside
        mask = dists > 1.0
        pos[mask] = pos[mask] / dists[mask, np.newaxis] * 0.999
    
    return pos
