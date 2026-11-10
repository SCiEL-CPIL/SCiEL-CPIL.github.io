import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    # Leave 10 seconds for safety
    deadline = start_time + 110.0

    # 1. Initialization: Phyllotaxis
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(N)
    r = np.sqrt(indices / N)
    theta = indices * golden_angle
    
    # Scale to fit inside
    # Max r is 1. We want to leave room for radius r_circle.
    # Initial guess for circle radius is ~ 0.05
    # So max center distance is 0.95
    x = r * 0.95 * np.cos(theta)
    y = r * 0.95 * np.sin(theta)
    centers = np.column_stack((x, y))
    
    best_centers = centers.copy()
    best_score = calculate_score(centers)
    
    step_size = 0.02
    min_step = 0.0001
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Calculate distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(centers, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Determine gradient
        # If pair distance is the bottleneck, use pair gradient
        # If boundary is the bottleneck, use boundary gradient
        # If both are tight, combine
        
        grad = np.zeros_like(centers)
        
        # 1. Pair Gradient (Top 5 closest pairs)
        flat_dists = dists.flatten()
        k = 5
        # Get indices of k smallest
        part_idx = np.argpartition(flat_dists, k)[:k]
        
        pair_grad = np.zeros_like(centers)
        for idx in part_idx:
            i = int(idx // N)
            j = int(idx % N)
            if i == j: continue
            d = dists[i, j]
            if d < 1e-6: continue
            vec = (centers[i] - centers[j]) / d
            pair_grad[i] += vec
            pair_grad[j] -= vec
            
        # 2. Boundary Gradient
        # If 1 - |c| is small, push inward
        boundary_grad = np.zeros_like(centers)
        # Points close to boundary
        close_to_boundary = norms > (1.0 - current_score - 0.01)
        for i in np.where(close_to_boundary)[0]:
            # Gradient of (1 - |c|) w.r.t c is -c/|c|
            # We want to maximize 1 - |c|, so move in direction -c
            boundary_grad[i] = -centers[i] / norms[i]
            
        # Combine
        # If pair distance is smaller (bottleneck), prioritize pair
        # If boundary is smaller, prioritize boundary
        # Weights:
        w_pair = 1.0 if (min_pair_dist / 2.0) <= min_boundary_dist else 0.5
        w_bound = 1.0 if min_boundary_dist <= (min_pair_dist / 2.0) else 0.5
        
        grad = w_pair * pair_grad + w_bound * boundary_grad
        
        # Normalize gradient to unit length to avoid huge steps
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-6:
            grad = grad / g_norm
            
        centers += step_size * grad
        
        # Hard constraint: Keep inside 1 - best_score/2 ?
        # No, keep inside 1 - current_score/2 to allow exploration?
        # Better: Keep inside 1 - 0.01 (safe margin)
        # Then the score calculation handles the rest.
        # But if we go outside, score becomes invalid/negative?
        # The task says "A centre on or outside the container ... invalid".
        # So we MUST stay inside.
        # We must ensure |c_i| < 1.
        # Let's project to 0.99 to be safe.
        norms_new = np.linalg.norm(centers, axis=1)
        outside = norms_new > 0.99
        if np.any(outside):
            centers[outside] = centers[outside] * (0.99 / norms_new[outside, np.newaxis])
            
        # Decay step size
        step_size = max(min_step, step_size * 0.999)
        
    return best_centers

def calculate_score(centers):
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair, min_bound)
