import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Approximate radius of circles in hex packing: r ~ 0.053
    # Spacing s = 2r ~ 0.1
    # Number of points ~ pi * (1/s)^2 * (2/si) ... 
    # Let's just generate a dense lattice and pick N points closest to center.
    
    s = 0.1
    # Generate lattice points
    points = []
    # Range of x, y
    # sqrt(3)/2 * s is vertical spacing
    # s is horizontal spacing for same row, offset by s/2 for next row
    
    max_k = int(1.0 / s) + 5
    for j in range(-max_k, max_k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    # Sort by radius to get central points
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii, kind="stable")
    c = points[indices[:N]].copy()
    
    # Ensure we have exactly N points
    if len(c) < N:
        # If lattice is too sparse, add random points
        while len(c) < N:
            c = np.vstack([c, np.random.uniform(-1, 1, 2)])
    
    # 2. Optimization: Gradient Ascent on Soft-Min
    alpha = 20.0  # Sharpness parameter
    eta = 0.005   # Step size
    
    best_radius = 0.0
    best_c = c.copy()
    
    # Pre-compute indices for pairs
    # We need to compute pairwise distances efficiently
    # Using broadcasting creates an NxN matrix, which is 300x300 = 90k elements. This is fine.
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        # Compute pairwise differences
        # c is (N, 2)
        # diff is (N, N, 2)
        # This might be memory heavy if N is large, but 300 is small.
        # 300 * 300 * 2 * 8 bytes = 1.44 MB. Very safe.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal (self distance)
        np.fill_diagonal(dist, np.inf)
        
        # Compute S_ij = exp(-alpha * dist)
        # To avoid underflow/overflow issues with large alpha, we can normalize or just let it be.
        # With alpha=20 and dist~0.1, exp(-2) is fine.
        S_ij = np.exp(-alpha * dist)
        
        # Compute S_i for boundary
        r_i = np.linalg.norm(c, axis=1)
        dist_boundary = 1.0 - r_i
        # If r_i > 1, dist_boundary is negative. exp(-alpha * negative) = exp(positive) -> large.
        # This creates a strong repulsive force from the boundary if outside.
        S_i = np.exp(-alpha * dist_boundary)
        
        # Sum for Z
        Z_ij = np.sum(S_ij)
        Z_i = np.sum(S_i)
        Z = Z_ij + Z_i
        
        if Z == 0 or not np.isfinite(Z):
            break
            
        # Gradients
        # grad_k = (alpha / Z) * [ sum_j S_kj * (c_k - c_j) / dist_kj - S_k * c_k / r_k ]
        
        # Term 1: sum_j S_kj * (c_k - c_j) / dist_kj
        # diff is (N, N, 2). dist is (N, N).
        # We want to sum over j (axis 1).
        # factor_kj = S_kj / dist_kj
        # factor is (N, N)
        factor = S_ij / dist
        # Set diagonal to 0
        np.fill_diagonal(factor, 0)
        
        # force_ij = factor[:, :, np.newaxis] * diff
        # force is (N, N, 2)
        force_ij = factor[:, :, np.newaxis] * diff
        
        # Sum over j (axis 1)
        grad_pair = np.sum(force_ij, axis=1) # Shape (N, 2)
        
        # Term 2: S_k * c_k / r_k
        # r_i is (N,)
        # c is (N, 2)
        # Avoid division by zero
        r_safe = np.where(r_i > 1e-8, r_i, 1e-8)
        grad_boundary = S_i[:, np.newaxis] * c / r_safe[:, np.newaxis]
        
        # Total Gradient
        grad = (alpha / Z) * (grad_pair - grad_boundary)
        
        # Update
        c += eta * grad
        
        # Clip to unit circle (safety)
        r_new = np.linalg.norm(c, axis=1)
        mask = r_new > 1.0
        if np.any(mask):
            c[mask] = c[mask] / r_new[mask][:, np.newaxis]
            
        # Calculate current radius
        # Min pair distance / 2
        # Min boundary distance
        min_pair_dist = np.min(dist[dist < np.inf])
        min_boundary_dist = np.min(1.0 - r_new)
        current_radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_c = c.copy()
            
        # Adaptive step size?
        # If radius decreases, reduce eta.
        # If radius increases, keep or slightly increase.
        # Simple: decay eta slowly
        eta *= 0.999
        
    return best_c
