"""Rewritten packing program using multi-start force-directed optimization with strict constraint projection."""
import numpy as np
import time

N = 300


def _compute_score(points):
    """Compute the packing radius score for a set of points."""
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    boundary = 1.0 - norms
    min_boundary = np.min(boundary)
    
    # If any point is outside, score is 0 (or very low)
    if min_boundary < 0:
        return 0.0

    # Pairwise distances
    # Use a chunked approach to avoid memory issues
    min_pair = np.inf
    n = len(points)
    # Vectorized pairwise distance calculation for efficiency
    # We can't compute full NxN matrix for N=300 easily in memory? 300x300 is small.
    # 300*300 = 90,000 floats. That's fine.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Mask out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)


def _init_hexagonal(n, seed=0):
    """Generate an initial hexagonal lattice configuration clipped to unit disk."""
    np.random.seed(seed)
    # Estimate spacing for n circles in unit disk
    r_est = np.sqrt(np.pi / (n * 2 * np.sqrt(3)))
    spacing = 2 * r_est
    
    # Generate hexagonal lattice points
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3) / 2
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if x * x + y * y <= 0.95: # Slightly inside to avoid boundary issues initially
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: random points in disk
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 0.95, n))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    else:
        # Select n points. Random selection from valid lattice points.
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    
    return points


def _optimize(points, deadline):
    """Optimize a single configuration using force-directed method."""
    n = len(points)
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Parameters
    lr = 0.01
    decay = 0.999
    iterations = 0
    max_iter = 5000
    
    # Precompute pair indices for efficiency? No, just loop.
    # Force: Repulsion between close pairs, Attraction to center? 
    # Actually, we want to maximize min distance.
    # Gradient of min distance is complex. 
    # Instead, use a potential energy approach: Minimize sum of 1/d^2 for close pairs?
    # Or simpler: Gradient ascent on min distance using soft min.
    
    # Let's use a simpler heuristic: 
    # 1. Find the closest pair (i, j).
    # 2. Push them apart.
    # 3. Find the point closest to boundary.
    # 4. Push it in.
    
    # This is a "local search" approach.
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find closest pair
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        if i == j:
            break
            
        dist_ij = dists[i, j]
        
        # Find point closest to boundary
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        boundary_dists = 1.0 - norms
        min_boundary_idx = np.argmin(boundary_dists)
        
        # Move i and j apart
        if dist_ij > 1e-8:
            dir_vec = (points[j] - points[i]) / dist_ij
            move_amount = lr * 0.5
            points[i] -= dir_vec * move_amount
            points[j] += dir_vec * move_amount
        else:
            # If coincident, move randomly
            points[i] += np.random.normal(0, 0.01, 2)
            points[j] += np.random.normal(0, 0.01, 2)
            
        # Move boundary point in
        if boundary_dists[min_boundary_idx] < 0.01:
            p = points[min_boundary_idx]
            norm_p = np.linalg.norm(p)
            if norm_p > 1e-8:
                points[min_boundary_idx] = p * (1.0 - lr)
            else:
                points[min_boundary_idx] = np.random.normal(0, 0.01, 2)
                
        # Project all points back into unit disk
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        mask = norms > 0.9999
        if np.any(mask):
            points[mask] *= (0.9999 / norms[mask, np.newaxis])
            
        # Compute score
        if iteration % 10 == 0:
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        # Decay learning rate
        lr *= decay
        
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Multi-start: Try several initializations
    # 1. Hexagonal lattice
    # 2. Random points
    # 3. Rotated hexagonal
    
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        # Different initializations
        if i == 0:
            points = _init_hexagonal(N, seed=42)
        elif i == 1:
            points = _init_hexagonal(N, seed=123)
        elif i == 2:
            # Random
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.9, N))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Perturbed hexagonal
            points = _init_hexagonal(N, seed=42)
            points += np.random.normal(0, 0.05, points.shape)
            # Project back
            norms = np.sqrt(np.sum(points ** 2, axis=1))
            mask = norms > 0.9999
            if np.any(mask):
                points[mask] *= (0.9999 / norms[mask, np.newaxis])
                
        # Optimize
        current_deadline = min(deadline - 2.0, start_time + (i + 1) * time_per_start)
        opt_points, opt_score = _optimize(points, current_deadline)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
            
    # Final check
    if best_points is None:
        # Fallback: just return hexagonal lattice
        best_points = _init_hexagonal(N, seed=42)
        
    # Ensure all points are strictly inside
    norms = np.sqrt(np.sum(best_points ** 2, axis=1))
    mask = norms > 0.99999
    if np.any(mask):
        best_points[mask] *= (0.99999 / norms[mask, np.newaxis])
        
    return best_points
