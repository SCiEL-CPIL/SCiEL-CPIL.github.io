# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_area_and_grad(x):
    """
    Compute the area of the union of 64 triangles and its subgradient with respect to x.
    x: shape (64,)
    Returns: area (float), grad (shape (64,))
    """
    n = 64
    # Quadrature points
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    # L_i(y) = x_i + a_i * y
    # Shape: (64, 20000)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n  # Shape: (20000,)
    
    # Sort L along axis 0 for each y
    L_sorted = np.sort(L, axis=0)
    
    # Gaps between consecutive sorted L's
    gaps = np.diff(L_sorted, axis=0)  # Shape: (63, 20000)
    
    # Union length contribution
    # Union length = w + sum(min(gaps, w))
    # We need the gradient of this sum with respect to x.
    
    # min(gap, w) is differentiable almost everywhere.
    # If gap < w, min(gap, w) = gap. Gradient w.r.t L_j and L_{j+1} is +1 and -1 respectively.
    # If gap >= w, min(gap, w) = w. Gradient is 0.
    # Note: gap = L_{j+1} - L_j.
    
    # Identify active gaps (where gap < w)
    # gap < w  <=>  L_{j+1} - L_j < w
    active = gaps < w[None, :]  # Shape: (63, 20000)
    
    # The contribution to the gradient from the sum of active gaps:
    # d/dx sum_{j} min(L_{j+1} - L_j, w)
    # = sum_{j} [ 1_{active_j} * (dL_{j+1}/dx - dL_j/dx) ]
    
    # We need to map the sorted indices back to the original x indices.
    # However, we can compute the gradient by accumulating contributions based on the sorted order.
    # Let order[y] be the permutation such that L_sorted[:, y] = L[order[:, y], y].
    # Actually, np.sort returns values, not indices. We need indices.
    
    # Re-do with indices for gradient computation
    order = np.argsort(L, axis=0)  # Shape: (64, 20000)
    
    # grad is shape (64,)
    grad = np.zeros(n)
    
    # For each active gap j at height y:
    # The term is (L_{sorted[j+1]} - L_{sorted[j]})
    # d/dx_k (L_{sorted[j+1]} - L_{sorted[j]}) = delta_{k, orig_idx(sorted[j+1])} - delta_{k, orig_idx(sorted[j])}
    
    # We can vectorize this accumulation.
    # For each column y:
    # active_j is True for j in 0..62.
    # If active_j is True:
    #   grad[order[j+1, y]] += 1
    #   grad[order[j, y]]   -= 1
    
    # Using np.add.at for efficient accumulation
    # We need to create arrays of indices and values to add.
    
    # Flatten active mask to get indices of active gaps
    # active shape: (63, 20000)
    # We want to add +1 to grad[idx_up] and -1 to grad[idx_down] for each active gap.
    
    # Get indices of active gaps
    # active_indices = np.where(active) -> (rows, cols)
    # rows are the gap index j (0..62), cols are the y index.
    
    # For each active gap (j, y):
    # idx_down = order[j, y]
    # idx_up = order[j+1, y]
    
    # We can construct the full list of updates.
    # This might be memory heavy (63*20000 ~ 1.2M updates), but manageable.
    
    # Alternative: Use bincount or add.at with pre-computed indices.
    
    # Let's try to vectorize the gradient calculation without explicit loops over y.
    # We can compute the gradient contribution for each x_i directly.
    
    # grad_i = sum_y [ (number of times x_i is the upper bound of an active gap at y) - (number of times x_i is the lower bound of an active gap at y) ]
    
    # Let's use np.add.at which is safe for repeated indices.
    # We need to generate the indices to update.
    
    # active_mask: (63, 20000)
    # For each y, we have a set of active gaps.
    
    # Let's iterate over the 63 gaps? No, that's slow in Python.
    # Let's use the fact that we can flatten.
    
    # active_flat = active.ravel()
    # rows_flat, cols_flat = np.where(active)
    
    # idx_down_flat = order[rows_flat, cols_flat]
    # idx_up_flat = order[rows_flat + 1, cols_flat]
    
    # np.add.at(grad, idx_down_flat, -1.0)
    # np.add.at(grad, idx_up_flat, 1.0)
    
    # This is O(N * M) memory for the indices, which is 1.2M * 2 * 8 bytes ~ 20MB. Acceptable.
    
    rows, cols = np.where(active)
    
    if len(rows) > 0:
        idx_down = order[rows, cols]
        idx_up = order[rows + 1, cols]
        
        np.add.at(grad, idx_down, -1.0)
        np.add.at(grad, idx_up, 1.0)
        
        # The term 'w' in union_len = w + sum(...) has 0 gradient.
        # The term 'w' in min(gap, w) when gap >= w has 0 gradient.
        
    # Calculate Area
    # union_len = w + sum(min(gaps, w))
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    area = union_len.mean()
    
    # Scale gradient by 1/20000 (mean)
    grad /= 20000.0
    
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initialize with Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    best_x = x.copy()
    best_area, _ = _calc_area_and_grad(best_x)
    
    # Subgradient Descent with Backtracking Line Search
    # Learning rate decay
    lr = 0.01
    min_lr = 1e-6
    
    # We will run for a fixed number of iterations or until time runs out.
    # Subgradient descent can be noisy, so we use a simple averaging or just take the best point.
    # Since the function is convex, the last point is often good, but tracking best is safer.
    
    # To handle the noise, we can use a larger initial step and decay it.
    # Or use Polyak step size if we have a lower bound, but we don't.
    # Let's use a standard decaying step size.
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        area, grad = _calc_area_and_grad(x)
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
            
        # Check for convergence
        if np.linalg.norm(grad) < 1e-8:
            break
            
        # Backtracking Line Search to find a step size that decreases the objective
        # We want to find alpha such that f(x - alpha * grad) < f(x)
        # Since subgradient is not exact gradient, we might not always decrease.
        # But for convex functions, it generally works well with small steps.
        # Let's try a fixed step with decay first, it's simpler and robust.
        # Step: x_new = x - lr * grad
        
        x_new = x - lr * grad
        
        # Optional: Clip x to a reasonable range to prevent divergence
        # The positions shouldn't need to be huge.
        # x_new = np.clip(x_new, -10, 10)
        
        # Check if this step improves the objective (or at least doesn't worsen it too much)
        # To save time, we might not evaluate every step if we trust the subgradient.
        # But evaluating is cheap (~5ms). 5000 * 5ms = 25s. We have 600s.
        # So we can afford to evaluate.
        
        new_area, _ = _calc_area_and_grad(x_new)
        
        if new_area < area:
            x = x_new
            # Decay learning rate slightly
            lr *= 0.99
        else:
            # If it didn't improve, reduce learning rate more aggressively
            lr *= 0.5
            # Optionally, try a smaller step in the same direction
            # For simplicity, just reduce lr.
            
        if lr < min_lr:
            lr = min_lr
            
    # Final check
    final_area, _ = _calc_area_and_grad(best_x)
    if final_area < best_area:
        best_area = final_area
        best_x = x.copy() # x might be better if last step was good? No, best_x tracks the min seen.
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        # Fallback to initial
        return np.array(positions, dtype=np.float64)
        
    return best_x

# EVOLVE-BLOCK-END
