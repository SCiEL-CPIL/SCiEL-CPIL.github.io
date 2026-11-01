import numpy as np
import time

N = 100
DEADLINE = 110.0  # Safety margin from 120s

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    # Boundary constraints
    # Distance to walls: 0.5 - max(abs(coord))
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    
    # Pairwise distances
    # To save memory and time, we compute pairwise distances in chunks or use cdist
    # For N=100, full matrix is 100x100 = 10000 floats, which is fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    r_pair = np.min(dist) / 2.0
    
    return min(r_wall, r_pair)

def _gradient_penalty(centers, r_target):
    """
    Compute the gradient of the penalty function for fixed r_target.
    Penalty = sum_{i<j} max(0, 2*r_target - d_ij)^2 + sum_{i,k} max(0, |c_ik| - (0.5 - r_target))^2
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    # Vectorized computation of distances and gradients
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # We only care about pairs where dist < 2*r_target
    # factor = max(0, 2*r - dist)
    factor = np.maximum(0.0, 2.0 * r_target - dist)
    
    # Gradient contribution for pair (i, j) on center i:
    # d/dc_i [ (2r - d)^2 ] = 2(2r - d) * d(2r - d)/dc_i
    # d(2r - d)/dc_i = - d(d)/dc_i = - (c_i - c_j) / d
    # So term = -2 * factor * (c_i - c_j) / d
    # Note: factor is 0 if dist >= 2r.
    
    # grad[i] += sum_j -2 * factor[i,j] * (centers[i] - centers[j]) / dist[i,j]
    # Note: factor[i,i] is 0 because dist[i,i]=0 -> factor=2r, but we set dist=1, 
    # actually we should mask diagonal properly.
    # Let's zero out diagonal in factor
    np.fill_diagonal(factor, 0.0)
    
    # Weight matrix for gradient
    # w[i,j] = -2 * factor[i,j] / dist[i,j]
    # We need to avoid div by zero if dist is 0 (coincident spheres).
    # If dist is 0, gradient is undefined, but penalty is huge. 
    # We can add a small epsilon to dist in denominator.
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    w = -2.0 * factor / dist_safe
    
    # grad[i] = sum_j w[i,j] * (centers[i] - centers[j])
    # This is equivalent to:
    # grad = sum_j (w[i,j] * centers[i] - w[i,j] * centers[j])
    # Let S = sum_j w[i,j]
    # grad[i] = S[i] * centers[i] - sum_j w[i,j] * centers[j]
    
    S = np.sum(w, axis=1)
    # sum_j w[i,j] * centers[j] is w @ centers
    weighted_sum = w @ centers
    
    grad += S[:, np.newaxis] * centers - weighted_sum
    
    # Wall penalties
    # Penalty: sum_{i,k} max(0, |c_ik| - (0.5 - r_target))^2
    # Let limit = 0.5 - r_target
    limit = 0.5 - r_target
    abs_c = np.abs(centers)
    excess = np.maximum(0.0, abs_c - limit)
    
    # d/dc [ (|c| - limit)^2 ] = 2 (|c| - limit) * sign(c)
    # So grad_wall[i,k] = 2 * excess[i,k] * sign(centers[i,k])
    grad += 2.0 * excess * np.sign(centers)
    
    return grad

def construct():
    start_time = time.time()
    
    # 1. Initialization: BCC-like or Jittered Grid
    # Let's start with a grid and add small jitter to break symmetry
    np.random.seed(42)
    
    # Grid 5x5x4
    m_x, m_y, m_z = 5, 5, 4
    xs = (np.arange(m_x) + 0.5) / m_x - 0.5
    ys = (np.arange(m_y) + 0.5) / m_y - 0.5
    zs = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    # Create grid
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T
    
    # We have 100 points exactly.
    
    # Add small random jitter to escape local minima of the grid
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to ensure they start inside (roughly)
    centers = np.clip(centers, -0.49, 0.49)
    
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # 2. Optimization Loop
    # We will try to increase r iteratively.
    # For each target r, we run gradient descent to minimize penalty.
    
    r_step = 0.001
    max_iter_per_r = 50
    lr = 0.01
    
    while time.time() - start_time < DEADLINE:
        # Try to increase radius
        target_r = current_r + r_step
        
        # Check if target_r is physically possible (max possible is ~0.5)
        if target_r > 0.45:
            break
            
        # Gradient Descent to fit target_r
        success = False
        for _ in range(max_iter_per_r):
            if time.time() - start_time > DEADLINE:
                break
                
            grad = _gradient_penalty(centers, target_r)
            
            # Update centers
            centers -= lr * grad
            
            # Optional: Clip to cube boundaries to prevent flying out
            # centers = np.clip(centers, -(0.5 - target_r) - 0.1, (0.5 - target_r) + 0.1)
            
            # Check penalty
            # If penalty is small, we can accept
            # We can just compute the actual radius and see if it improved or if penalty is low
            # For efficiency, check penalty every few steps? 
            # Let's compute radius occasionally.
            
        # After gradient steps, check if we achieved target_r
        # We need to ensure spheres don't overlap and are inside.
        # The gradient descent minimizes the penalty for target_r.
        # If the minimum penalty is 0, then target_r is valid.
        
        # To be safe, compute the actual valid radius of the new centers
        new_r = _compute_radius(centers)
        
        if new_r > current_r - 1e-6: # Allowed slight numerical drift
            # Check if we actually reached close to target_r or if it's a valid improvement
            # If new_r is significantly larger than current_r, accept
            if new_r > current_r:
                current_r = new_r
                best_centers = centers.copy()
                best_r = current_r
                # Keep step size or increase slightly?
                # r_step = max(0.0001, r_step * 0.9) # Slow down if we are pushing hard?
                # Or keep constant.
            else:
                # If we didn't improve, maybe we are stuck.
                # Try a smaller step or kick
                pass
        else:
            # Failed to maintain even current radius? Revert
            # This shouldn't happen if penalty minimization works, but just in case
            pass
            
        # If we are stuck (no improvement in last few iterations), add a kick
        # Simple heuristic: if current_r hasn't changed in a while, perturb
        # We don't track 'no change' count here explicitly, but we can add periodic kicks
        
        # Periodic kick to escape local minima
        if int(time.time() - start_time) % 5 == 0: # Roughly every 5 seconds
            centers += np.random.normal(0, 0.005, centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            current_r = _compute_radius(centers)
            if current_r < best_r:
                # Revert to best if kick was bad? 
                # Actually, let the next gradient steps fix it. 
                # But if it's too bad, reset to best.
                if current_r < best_r * 0.9:
                    centers = best_centers.copy()
                    current_r = best_r

    return best_centers

# Ensure the function returns the best found
# The loop above updates best_centers whenever an improvement is found.
