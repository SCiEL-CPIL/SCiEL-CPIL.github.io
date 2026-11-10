import numpy as np
import time

N = 300

def construct():
    # 1. Initialize with hexagonal lattice
    # Approximate spacing for N=300 in unit circle
    # Area of unit circle = pi. Area of hex cell ~ sqrt(3)/2 * s^2.
    # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate hex lattice points
    points = []
    # Range to cover unit circle
    R = 1.5
    k = int(R / s_init) + 2
    for j in range(-k, k + 1):
        y = j * s_init * np.sqrt(3) / 2
        x_offset = (j % 2) * s_init / 2
        for i in range(-k, k + 1):
            x = i * s_init + x_offset
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback: random points
        points = np.random.randn(N, 2) * 0.5
    else:
        # Select N points closest to center or random? 
        # Random selection from valid lattice points might be better to avoid bias
        # But lattice is structured. Let's just take the first N or random N.
        # Actually, taking the densest part (center) is good.
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
    # Ensure finite and valid
    points = np.nan_to_num(points)
    
    best_x = points.copy()
    best_r = -np.inf
    
    # Calculate initial score
    def calc_score(x):
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)
    
    best_r = calc_score(best_x)
    
    start_time = time.time()
    max_time = 110.0 # Safety margin for 120s limit
    
    # Optimization loop
    # We use a gradient ascent approach on the "soft" minimum
    # To make it smoother, we consider the k-closest constraints
    
    alpha = 0.01 # Initial step size
    decay = 0.999 # Step decay
    k_closest = 10 # Number of closest pairs/points to consider for gradient
    
    while time.time() - start_time < max_time:
        # 1. Compute distances
        diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        bound_dists = 1.0 - np.linalg.norm(best_x, axis=1)
        
        # 2. Identify active constraints
        # We want to maximize r = min(min_pair/2, min_bound)
        # Let's look at the smallest distances
        flat_dists = dists.flatten()
        # Get indices of smallest k pair distances
        # Note: dists is symmetric, so we get duplicates. 
        # We can just work with the upper triangle or all.
        # Using all is simpler for vectorization, just double counts force, which is fine if we normalize.
        
        # Get k smallest pair distances
        # We need indices in 2D
        flat_indices = np.argsort(flat_dists)
        
        # We also need k smallest boundary distances
        bound_indices = np.argsort(bound_dists)
        
        # Determine current r
        min_pair_val = flat_dists[flat_indices[0]] / 2.0
        min_bound_val = bound_dists[bound_indices[0]]
        current_r = min(min_pair_val, min_bound_val)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            # No need to update best_x here if we are iterating on it, 
            # but we should keep a separate copy for the final return
            # Actually, we are modifying best_x in place. 
            # Let's use a working array `x` and `best_x` for the record.
            
        # Let's switch to working with `x` to allow backtracking if needed, 
        # or just greedy ascent. Greedy is usually fine for packing.
        
        # Compute Gradient
        grad = np.zeros_like(best_x)
        
        # Threshold for "active" constraints
        # We want to push apart pairs that are close to the minimum
        # and pull in points close to boundary.
        # Let's use a tolerance relative to current_r
        tol = 1e-4 + 0.1 * current_r
        
        # Pair forces
        # For each of the k_closest pairs
        for idx in flat_indices[:k_closest * 2]: # *2 because of symmetry
            i, j = np.unravel_index(idx, dists.shape)
            if i == j: continue
            d = dists[i, j]
            if d < current_r * 2 + tol * 2: # d is full distance, r is radius (d/2)
                # Force direction: from j to i (repel i, attract j? No, repel both)
                # Vector from j to i
                v = best_x[i] - best_x[j]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-8:
                    v = v / norm_v
                    # Strength proportional to how much it violates the "target" radius
                    # We want d >= 2*current_r
                    # If d < 2*current_r, push apart
                    # Force magnitude
                    f_mag = (2 * current_r - d) / (2 * current_r) # Normalized
                    # Or just constant push?
                    # Let's use a smooth function
                    # f = 1 if d < 2r, 0 otherwise?
                    # Gradient of min is 1 for the min element.
                    # So we add unit vector.
                    grad[i] += v
                    grad[j] -= v
        
        # Boundary forces
        for idx in bound_indices[:k_closest]:
            i = idx
            d_b = bound_dists[i]
            if d_b < current_r + tol:
                # Push inward
                v = -best_x[i]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-8:
                    v = v / norm_v
                    grad[i] += v
        
        # Normalize gradient to avoid huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
            
        # Update
        best_x += alpha * grad
        
        # Project back to unit circle if any point goes out?
        # The force should keep them in, but just in case:
        norms = np.linalg.norm(best_x, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            best_x[out_mask] /= norms[out_mask, np.newaxis]
            
        # Decay step size
        alpha *= decay
        if alpha < 1e-6:
            alpha = 1e-4 # Reset to explore? Or stop?
            # Let's keep it small to polish
            
    return best_x
