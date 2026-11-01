import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # 1. Initialization: Hexagonal lattice
    # We want roughly 300 points in a unit circle.
    # Area of unit circle = pi. Area per point = pi/300.
    # Hex lattice spacing 's' has area per point s * s * sqrt(3)/2.
    # s * s * sqrt(3)/2 = pi/300 => s = sqrt(2*pi / (300*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate points in a large grid
    k = int(1.0 / s) + 3
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            points_list.append([x, y])
    
    points = np.array(points_list)
    radii = np.hypot(points[:, 0], points[:, 1])
    # Keep points strictly inside to avoid initial boundary issues, 
    # but we will clip later. 
    # Actually, for packing, we want them as far as possible from center? No, we want to fill the space.
    # Let's just take all points and select the first N? 
    # Better: take points in a disk of radius 1.
    mask = radii <= 1.0
    points = points[mask]
    
    # If we have more than N, we need to select N.
    # If we have less, we need to add more or adjust s.
    # Let's adjust s to get close to N points.
    while len(points) < N:
        s *= 0.95
        points_list = []
        k = int(1.0 / s) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points_list.append([x, y])
        points = np.array(points_list)
        radii = np.hypot(points[:, 0], points[:, 1])
        points = points[radii <= 1.0]
        
    while len(points) > N:
        # Remove points with largest radius? Or random? 
        # Removing outer points might create holes. 
        # Let's just take the first N after sorting by radius to keep a disk shape?
        # Or better, reduce s slightly so we get fewer points? 
        # If s is too small, we have too many points. We should increase s.
        s *= 1.05
        points_list = []
        k = int(1.0 / s) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points_list.append([x, y])
        points = np.array(points_list)
        radii = np.hypot(points[:, 0], points[:, 1])
        points = points[radii <= 1.0]
        
    # If we still don't have exactly N, we might have a count mismatch.
    # The task requires exactly N points.
    # If len(points) != N, we need to handle it.
    # Usually, adjusting s will hit N or pass it. 
    # If we pass, we take the first N (closest to center? or random?).
    # Let's sort by distance to center and take the first N to ensure a compact cluster.
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radii)[:N]]
    elif len(points) < N:
        # This shouldn't happen if we looped enough, but just in case:
        # Add random points? 
        # Better to restart with a slightly larger s? 
        # For safety, if we can't get exactly N, we might fail. 
        # But the loop above should converge.
        pass

    if len(points) != N:
        # Fallback: generate random points if lattice fails to hit N
        points = np.random.rand(N, 2) * 2 - 1
        radii = np.hypot(points[:, 0], points[:, 1])
        points[radii > 1] = points[radii > 1] / radii[radii > 1, None]
        
    # 2. Optimization: Gradient Descent on Energy E = sum(1/d^2)
    # This maximizes the minimum distance effectively.
    
    current_points = points.copy()
    best_score = -1.0
    best_points = current_points.copy()
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    step_size = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
            
        # Calculate distances
        diffs = current_points[i_idx] - current_points[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Avoid division by zero
        dists_safe = np.maximum(dists, 1e-6)
        
        # Force magnitude: gradient of 1/d^2 is -2/d^3 * (dx/dx_i)
        # Force on i from j: F_ij = 2/d^3 * (c_i - c_j)
        # We want to minimize E, so we move in direction -grad E.
        # grad E / dx_i = sum_j 2/d_ij^3 * (c_i - c_j)
        # So update: c_i -= step * grad
        
        forces = np.zeros_like(current_points)
        # Vectorized force accumulation
        # factor = 2 / dists_safe**3
        # contributions = factor[:, None] * diffs
        # np.add.at(forces, i_idx, contributions)
        # np.add.at(forces, j_idx, -contributions)
        
        # This is faster in a loop if N is small, but N=300 is okay for vectorized.
        # However, np.add.at is slow. Let's use bincount or scatter.
        # Or just a simple loop over pairs? 300*300/2 = 45000 pairs. 
        # A Python loop over 45k pairs is ~0.01-0.05s. 
        # 5000 iterations * 0.05s = 250s. Too slow.
        # Must use vectorized numpy.
        
        # Vectorized approach:
        # Create a matrix of forces? No, sparse.
        # Let's use np.add.at for simplicity first, check speed.
        # Actually, we can compute the gradient using matrix operations if we build the full distance matrix?
        # D = c[:, None, :] - c[None, :, :]
        # Dist = norm(D, axis=2)
        # Grad = 2 * D / Dist**3 (with diagonal 0)
        # Sum over axis 1 to get force on each point.
        
        D = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists_mat = np.linalg.norm(D, axis=2)
        # Set diagonal to 1 to avoid div by 0, then zero out
        np.fill_diagonal(dists_mat, 1.0)
        # Factor 2 / d^3
        factors = 2.0 / (dists_mat ** 3)
        np.fill_diagonal(factors, 0.0)
        
        # Force on i is sum_j factors[i,j] * D[i,j]
        # F_i = sum_j (2/d^3) * (c_i - c_j)
        # This is exactly: np.sum(factors[:, :, np.newaxis] * D, axis=1)
        forces = np.sum(factors[:, :, np.newaxis] * D, axis=1)
        
        # Update
        current_points -= step_size * forces
        
        # Clip to unit circle
        radii = np.linalg.norm(current_points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            current_points[mask] /= radii[mask, np.newaxis]
            
        # Calculate actual score
        # Min pair distance / 2
        # Min boundary distance
        pair_dists = dists_mat[i_idx, j_idx] # Reuse from above? No, D changed.
        # Recalculating dists is expensive. 
        # We can estimate score from forces? No.
        # Let's calculate score every 10 iterations to save time.
        
        if it % 10 == 0:
            # Pairwise distances
            d = np.linalg.norm(current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d) / 2.0
            min_bound = np.min(1.0 - np.linalg.norm(current_points, axis=1))
            score = min(min_pair, min_bound)
            
            if score > best_score:
                best_score = score
                best_points = current_points.copy()
                
            # Adaptive step size?
            # If score improved, keep step. If not, reduce step?
            # For now, fixed step.
            
        # Decay step size slowly?
        if it > 1000:
            step_size *= 0.999

    return best_points

# EVOLVE-BLOCK-END
