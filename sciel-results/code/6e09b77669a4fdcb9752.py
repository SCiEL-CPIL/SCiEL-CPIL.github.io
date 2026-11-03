"""
Optimized circle packing for N=300 using vectorized gradient ascent on a soft-min objective.
"""
import numpy as np
import time

N = 300

def _init_hex_lattice(n_points):
    """
    Initialize points using a hexagonal lattice cut to the unit disk.
    We generate a large enough lattice, filter points inside the unit disk,
    and select the n_points closest to the center to form a compact cluster.
    """
    # Estimate spacing
    # Approximate radius r ~ 0.055
    # Spacing s ~ 2*r ~ 0.11
    # To be safe, start with smaller spacing and filter
    spacing = 0.1
    
    # Generate grid
    k = int(1.0 / spacing) + 5
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    if not points:
        return np.zeros((n_points, 2))
        
    points = np.array(points)
    # Sort by distance to center, take first n_points
    points = points[np.argsort(points[:, 2])]
    
    if len(points) >= n_points:
        return points[:n_points, :2]
    else:
        # If not enough points (should not happen with k large enough), pad with random
        remaining = n_points - len(points)
        random_pts = np.random.uniform(-0.9, 0.9, size=(remaining, 2))
        return np.vstack([points[:, :2], random_pts])

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    points = _init_hex_lattice(N)
    
    # Ensure valid initial state (projection)
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        points[mask] = points[mask] / norms[mask, np.newaxis]
        
    best_score = -1.0
    best_points = points.copy()
    
    # Optimization parameters
    beta = 20.0  # Soft-min sharpness
    lr = 0.005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    t = 0
    
    # Pre-allocate indices for pairwise calculations
    # i, j indices for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    while time.time() - start_time < timeout:
        t += 1
        
        # 1. Compute distances and boundary terms
        # Pairwise distances
        diffs = points[i_idx] - points[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - norms
        
        # 2. Form objective terms X
        # We want to maximize min(dists/2, boundary_dists)
        # Terms: dists/2 for pairs, boundary_dists for points
        x_pairs = dists_safe / 2.0
        x_bound = boundary_dists
        
        # Combine into single array for soft-min
        # To save memory and time, we can compute weights separately
        # But soft-min requires global normalization.
        # Let's compute log-sum-exp style for stability
        
        # x = concat(x_pairs, x_bound)
        # We need weights w_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
        
        # To avoid overflow/underflow, subtract max
        max_x = np.max(np.concatenate([x_pairs, x_bound]))
        
        exp_pairs = np.exp(-beta * (x_pairs - max_x))
        exp_bound = np.exp(-beta * (x_bound - max_x))
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bound)
        
        w_pairs = exp_pairs / sum_exp
        w_bound = exp_bound / sum_exp
        
        # 3. Compute Gradient
        # grad = sum w_k * grad(x_k)
        
        # Gradient of x_pairs[k] = |c_i - c_j| / 2
        # d/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
        # d/dc_j = -(c_i - c_j) / (2 * |c_i - c_j|)
        
        # Unit vectors
        # diffs is (num_pairs, 2)
        # dists_safe is (num_pairs,)
        unit_vecs = diffs / dists_safe[:, np.newaxis]
        
        # Weighted gradients for pairs
        # w_pairs is (num_pairs,)
        # grad_pairs_i = w_pairs * unit_vecs / 2
        # grad_pairs_j = - w_pairs * unit_vecs / 2
        
        grad = np.zeros_like(points)
        
        # Accumulate for i
        np.add.at(grad, i_idx, w_pairs[:, np.newaxis] * unit_vecs * 0.5)
        # Accumulate for j
        np.add.at(grad, j_idx, -w_pairs[:, np.newaxis] * unit_vecs * 0.5)
        
        # Gradient of x_bound[i] = 1 - |c_i|
        # d/dc_i = - c_i / |c_i|
        # Handle c_i = 0
        norms_safe = np.where(norms < 1e-8, 1e-8, norms)
        grad_bound = -points / norms_safe[:, np.newaxis]
        
        # Weighted gradient for boundary
        grad += w_bound[:, np.newaxis] * grad_bound
        
        # 4. Adam Update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        points = points + lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # 5. Projection to ensure validity
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] = points[mask] / norms[mask, np.newaxis]
            
        # 6. Calculate Score
        # Min of min(pair_dist/2) and min(boundary_dist)
        min_pair_dist = np.min(dists_safe) / 2.0
        min_bound_dist = np.min(boundary_dists)
        score = min(min_pair_dist, min_bound_dist)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
        # 7. Schedule
        # Increase beta to sharpen minimum
        if t % 100 == 0:
            beta *= 1.1
            if beta > 100:
                beta = 100
            # Decay learning rate
            lr *= 0.95
            if lr < 1e-5:
                lr = 1e-5
                
        # Check time
        if time.time() - start_time > timeout - 5:
            break
            
    # Final verification
    # Ensure no NaNs
    if not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    # Ensure inside circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis]
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
