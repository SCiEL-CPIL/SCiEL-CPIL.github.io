import numpy as np
import time
import math

N = 100

def energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    points: (N, 3) array of unit vectors.
    Returns:
        energy: float
        grad: (N, 3) array
    """
    # Calculate pairwise distances
    # dist2[i, j] = ||points[i] - points[j]||^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    # We only sum over i < j, so we can mask the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # We compute the full sum and divide by 2
    # log(d) = 0.5 * log(dist2)
    # So energy = -0.5 * sum_{i != j} log(dist2[i,j]) / 2 ? No.
    # sum_{i<j} log(d_ij) = 0.5 * sum_{i != j} log(d_ij)
    # E = - sum_{i<j} log(d_ij) = -0.5 * sum_{i != j} log(d_ij)
    
    # Clip dist2 to avoid log(0)
    dist2_clipped = np.maximum(dist2, 1e-12)
    
    log_dist2 = np.log(dist2_clipped)
    # Sum over all i != j
    total_log_dist2 = np.sum(log_dist2) - N * np.log(1.0) # diagonal is log(1)=0
    energy = -0.5 * total_log_dist2
    
    # Gradient:
    # grad_i = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    # Let's compute this efficiently.
    # diff[i, j] = x_i - x_j
    # We need sum_j (x_j - x_i) / dist2[i, j]
    # Note: diff is x_i - x_j, so x_j - x_i = -diff
    # grad_i = - sum_j diff[i, j] / dist2[i, j]
    
    # dist2 is (N, N)
    # We need to divide each row by dist2 row-wise, but only for j != i
    # Since dist2[i, i] is 1.0 (masked), and diff[i, i] is 0, the term is 0.
    # So we can just divide the whole matrix.
    
    inv_dist2 = 1.0 / dist2_clipped
    # Multiply diff by inv_dist2 element-wise
    # diff is (N, N, 3)
    # inv_dist2 is (N, N) -> (N, N, 1)
    weighted_diff = diff * inv_dist2[:, :, np.newaxis]
    
    # grad_i = - sum_j weighted_diff[i, j]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Gradient descent with momentum
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(points)
    
    best_energy = np.inf
    best_points = points.copy()
    
    # First, calculate initial energy
    e0, g0 = energy_and_grad(points)
    best_energy = e0
    
    max_iter = 5000
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Update velocity
        v = momentum * v - lr * g0
        
        # Proposed new points
        new_points = points + v
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Calculate new energy
        e1, g1 = energy_and_grad(new_points)
        
        # Backtracking line search if energy increased
        if e1 > best_energy:
            # Reduce learning rate
            lr *= 0.5
            # Retry with smaller step
            # Re-calculate v with new lr? Or just scale the step?
            # Let's just scale the step taken
            # We already updated v, so let's revert and try smaller step
            # Actually, simpler: just reduce lr and continue. The momentum might be messed up.
            # Better: use a simple line search on the direction.
            # For now, just reduce lr and hope momentum doesn't hurt too much.
            # Or, reset momentum if step fails?
            # Let's try: if e1 > e0, reduce lr and try again with same v?
            # No, let's just reduce lr and continue.
            continue
            
        # Accept step
        points = new_points
        g0 = g1
        if e1 < best_energy:
            best_energy = e1
            best_points = points.copy()
            
        # Decay learning rate slowly
        lr = max(0.001, lr * 0.999)
        
    return best_points
