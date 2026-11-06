"""Rewritten program for packing 200 equal circles in a unit square.
Uses force-based optimization with simultaneous repulsion from all pairs
and wall constraints, with multiple restarts from diverse initial grids.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 148.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0

    # Distance to walls: min_i min_k (0.5 - |c_ik|)
    dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                    np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
    wall_term = np.min(dw)

    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_term)

    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0

    return min(float(wall_term), float(pair_term))


def _hexagonal_grid(n_circles, row_count=None, seed=42):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    rng = np.random.RandomState(seed)
    
    if row_count is None:
        # Estimate optimal row count
        row_count = int(np.sqrt(n_circles / np.sqrt(3))) + 1
        row_count = max(8, min(30, row_count))
    
    # Distribute N circles into row_count rows as evenly as possible
    base_per_row = n_circles // row_count
    extra = n_circles % row_count
    
    row_counts = []
    for i in range(row_count):
        if i < extra:
            row_counts.append(base_per_row + 1)
        else:
            row_counts.append(base_per_row)
    
    max_row_count = max(row_counts)
    
    # Compute spacing to fit in unit square
    r_width = 1.0 / (2.0 * max_row_count)
    r_height = 1.0 / ((row_count - 1) * np.sqrt(3) + 2)
    r_est = min(r_width, r_height)
    
    dx = 2.0 * r_est
    dy = np.sqrt(3) * r_est
    
    centers = []
    y_offset = -0.5 + r_est
    
    for row_idx, row_count in enumerate(row_counts):
        x_offset = 0.0
        if row_idx % 2 == 1:
            x_offset = dx / 2.0
        
        if row_count > 0:
            row_width = (row_count - 1) * dx
            start_x = -row_width / 2.0 + x_offset
        else:
            start_x = 0.0
        
        for col_idx in range(row_count):
            x = start_x + col_idx * dx
            y = y_offset + row_idx * dy
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the arrangement
    centers[:, 0] -= np.mean(centers[:, 0])
    centers[:, 1] -= np.mean(centers[:, 1])
    
    # Clip to be within bounds
    centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return centers


def _optimize_forces(centers, time_budget=30.0):
    """
    Force-based optimization: compute repulsive forces from all nearby pairs
    and wall repulsion, then move all circles simultaneously.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()
    
    # Force parameters
    cutoff = 0.15  # Only consider pairs within this distance
    wall_margin = 0.1  # Wall force active within this distance of boundary
    
    step_size = 0.005
    min_step = 0.0001
    cool_rate = 0.9995
    
    start_t = time.time()
    no_improve_count = 0
    patience = 500
    
    while _remaining_time() > 5.0 and (time.time() - start_t) < time_budget:
        # Compute all pairwise distances
        d = pdist(centers)
        n = len(centers)
        
        # Initialize forces
        forces = np.zeros((n, 2))
        
        # Compute forces from all pairs within cutoff
        # Use a more efficient approach: only process pairs within cutoff
        cutoff2 = cutoff * cutoff
        
        for i in range(n - 1):
            # Distances from point i to points i+1, ..., n-1
            start_idx = i * n - i * (i + 1) // 2
            end_idx = start_idx + (n - i - 1)
            dists = d[start_idx:end_idx]
            
            # Find pairs within cutoff
            mask = dists < cutoff
            if not np.any(mask):
                continue
            
            indices = np.where(mask)[0]
            for local_idx in indices:
                j = i + 1 + local_idx
                dist = dists[local_idx]
                if dist < 1e-10:
                    # Coincident: random small push
                    vec = np.random.randn(2) * 0.001
                    forces[i] += vec
                    forces[j] -= vec
                    continue
                
                # Repulsive force: stronger when closer
                # Force magnitude ~ 1/dist^2 for strong repulsion at close range
                force_mag = 1.0 / (dist * dist)
                vec = centers[i] - centers[j]
                unit_vec = vec / dist
                forces[i] += force_mag * unit_vec
                forces[j] -= force_mag * unit_vec
        
        # Wall forces
        for i in range(n):
            x, y = centers[i]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            
            if dl < wall_margin:
                forces[i, 0] += 1.0 / (dl + 1e-6) ** 2
            if dr < wall_margin:
                forces[i, 0] -= 1.0 / (dr + 1e-6) ** 2
            if db < wall_margin:
                forces[i, 1] += 1.0 / (db + 1e-6) ** 2
            if dt < wall_margin:
                forces[i, 1] -= 1.0 / (dt + 1e-6) ** 2
        
        # Normalize forces to avoid huge steps
        force_norms = np.linalg.norm(forces, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 1.0:
            forces = forces / max_norm
        
        # Apply forces
        centers += step_size * forces
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Compute new radius
        new_r = _compute_radius(centers)
        
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > patience:
            step_size *= 0.9
            if step_size < min_step:
                # Try a random kick to escape local minimum
                if np.random.random() < 0.3:
                    idx = np.random.randint(n)
                    centers[idx] += np.random.randn(2) * 0.02
                    centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                no_improve_count = 0
        
        # Cool down step size
        step_size = max(min_step, step_size * cool_rate)
    
    return best_centers, best_r


def construct():
    """Construct optimal circle packing."""
    # Generate multiple initial configurations and optimize each
    best_radius = 0.0
    best_centers = None
    
    # Try different row counts for hexagonal grid
    row_counts_to_try = [12, 13, 14, 15, 16, 17, 18]
    seeds_to_try = [42, 123, 456, 789, 101112]
    
    # First pass: try various hexagonal grids with full optimization
    for row_count in row_counts_to_try:
        if _remaining_time() < 20.0:
            break
        
        centers = _hexagonal_grid(N, row_count=row_count, seed=42)
        r = _compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Optimize
        if _remaining_time() > 25.0:
            opt_centers, opt_r = _optimize_forces(centers, time_budget=20.0)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers
    
    # Second pass: if time remains, do additional optimizations from best found
    if _remaining_time() > 30.0 and best_centers is not None:
        # Try perturbing the best solution and re-optimizing
        for seed in seeds_to_try:
            if _remaining_time() < 15.0:
                break
            
            rng = np.random.RandomState(seed)
            perturbed = best_centers.copy()
            # Small random perturbation
            perturbed += rng.randn(N, 2) * 0.005
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            opt_centers, opt_r = _optimize_forces(perturbed, time_budget=12.0)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers
    
    # Final: ensure valid output
    if best_centers is None:
        best_centers = _hexagonal_grid(N, row_count=14, seed=42)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
