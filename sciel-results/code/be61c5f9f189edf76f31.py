import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing. Area of unit circle = pi. Area per circle ~ pi/r^2.
    # r ~ 0.053. Area per circle ~ pi / 0.053^2 ~ 1100.
    # Hex lattice density: sqrt(3)/2 * s^2.
    # s^2 ~ 1100 * 2/sqrt(3) ~ 1270. s ~ 35? No.
    # Let's just use a fine grid.
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    points = np.array(points)
    
    # If we have more than N points, keep the N closest to center? 
    # Or just take the first N? 
    # Taking closest to center is usually better for initial density.
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points if we didn't get enough (should not happen with spacing 0.08)
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        
    points = points[:N]
    
    # 2. Optimization Loop
    # We use a soft-min gradient ascent.
    # Objective: Maximize min(constraints)
    # Constraints: d_ij/2 for i<j, 1-|c_i| for i
    
    alpha = 0.05
    cooling = 0.999
    k_soft = 50.0  # Sharpness of soft-min
    
    best_score = 0.0
    best_points = points.copy()
    
    while time.time() - start_time < time_limit:
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        # Flatten pairwise distances / 2
        i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        # Check if improved
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
        
        # Compute Soft-Min Weights
        # Weights are exp(-k * (val - min))
        # To avoid overflow/underflow, subtract min first (which is 0 for the min element)
        # Actually, val - min >= 0.
        weights = np.exp(-k_soft * (all_vals - current_min))
        weights /= np.sum(weights)
        
        # Compute Gradients
        # Gradient of val w.r.t points
        # For pair (i,j) with value d/2:
        # grad_i = (c_i - c_j) / (2 * d)
        # grad_j = (c_j - c_i) / (2 * d)
        # For boundary i with value 1-|c_i|:
        # grad_i = -c_i / |c_i|
        
        grads = np.zeros_like(points)
        
        # Pair contributions
        # We need to map weights back to pairs
        pair_weights = weights[:len(pair_vals)]
        
        # Vectorized gradient calculation for pairs
        # diff[i_idx, j_idx] is c_i - c_j
        # dists[i_idx, j_idx] is d
        # grad_vec = diff / d
        # We need to accumulate into grads[i] and grads[j]
        
        # This is tricky to vectorize fully without loops or advanced indexing tricks.
        # Given N=300, a loop over pairs might be too slow if done in Python.
        # Let's try to vectorize the accumulation.
        
        # We have arrays i_idx, j_idx, pair_weights, diff, dists.
        # grad_i_part = pair_weights * (diff / dists)[:, None]
        # We need to sum grad_i_part for all j where i is the first index.
        # And sum -grad_i_part for all i where j is the first index? No.
        
        # Let's use np.add.at
        # grads[i] += sum_{j} w_{ij} * (c_i - c_j) / d_{ij}
        # grads[j] -= sum_{i} w_{ij} * (c_i - c_j) / d_{ij}  (since grad_j is negative of grad_i direction)
        
        # Note: diff = c_i - c_j.
        # grad_i = w * (c_i - c_j) / d
        # grad_j = w * (c_j - c_i) / d = - w * (c_i - c_j) / d
        
        # So we add w * (c_i - c_j) / d to grads[i]
        # And subtract w * (c_i - c_j) / d from grads[j]
        
        vecs = (diff[i_idx, j_idx] / dists[i_idx, j_idx, np.newaxis]) * pair_weights[:, np.newaxis]
        
        np.add.at(grads, i_idx, vecs)
        np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[len(pair_vals):]
        # grad_i = - c_i / |c_i| * w_i
        # rad > 0 usually.
        bound_grads = - (points / rad[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Update points
        points += alpha * grads
        
        # Project back inside circle? 
        # The gradient for boundary pushes inward, but we might overshoot.
        # Let's clip or project.
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cool down
        alpha *= cooling
        
        # Optional: Random perturbation to escape local minima
        if np.random.rand() < 0.1:
            points += np.random.normal(0, alpha * 0.5, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]

    return best_points
