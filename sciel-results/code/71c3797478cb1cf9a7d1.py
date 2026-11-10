import numpy as np
import time


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area using quadrature."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    return area


def _compute_area_fast(positions, a, y, w):
    """Faster area computation with precomputed constants."""
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    return area


def _subgradient(positions, a, y, w, n=64, n_heights=20000):
    """
    Compute subgradient of the union area with respect to positions.
    For each height, the union length is piecewise linear in each x_i.
    We determine which triangles are 'active' (on the union boundary) 
    and assign ±1 contributions accordingly.
    """
    n = len(positions)
    grad = np.zeros(n)
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    # For each height, determine which original indices are leftmost and rightmost
    # in the sorted order, and which are in the interior (covered by overlap)
    
    # Get the sorted indices
    sort_idx = np.argsort(L, axis=0)  # shape (n, n_heights)
    
    diffs = np.diff(L_sorted, axis=0)  # shape (n-1, n_heights)
    
    # A triangle at sorted position j is "exposed" on the left if it's the first,
    # and on the right if it's the last. Interior triangles contribute to overlap.
    # 
    # The union length = w + sum_{j=0}^{n-2} min(d_j, w)
    # where d_j = L_{j+1} - L_j (sorted)
    #
    # d/dx_i of L_{j+1} - L_j: if i is at sorted position j, contributes -1;
    # if at position j+1, contributes +1.
    # 
    # min(d_j, w) derivative: 1 if d_j < w, 0 if d_j > w, subgradient in [0,1] if d_j == w
    #
    # So the subgradient for position i at height m:
    #   = sum over j where i is at sorted position j: [-min(1, ...) + min(1, ...)]
    # More precisely: for each consecutive pair (j, j+1) in sorted order:
    #   if d_j < w: the pair contributes (1 if i is at pos j+1, -1 if at pos j, 0 else)
    #   if d_j >= w: no contribution from this pair
    # Plus: the base w contributes nothing to gradient.
    # But wait - we also need to account for the fact that only the "exposed" parts count.
    
    # Actually, let me think more carefully.
    # union_length = w + sum_j min(d_j, w)
    # d_j = L_{sorted[j+1]} - L_{sorted[j]}
    # d/dx_i min(d_j, w) = min(d_j, w)' * d(d_j)/dx_i
    # = (1 if d_j < w else 0) * (1 if i is at sorted pos j+1 else 0 - 1 if i is at sorted pos j else 0)
    
    # So for each height m:
    #   grad_i += sum_j [indicator(d_j < w) * (indicator(i at sorted pos j+1) - indicator(i at sorted pos j))]
    
    # This simplifies: for each height, the gradient contribution is:
    #   For the leftmost triangle in sorted order: -indicator(d_0 < w)
    #   For the rightmost: +indicator(d_{n-2} < w)  
    #   For interior at sorted pos k (0 < k < n-1): +indicator(d_{k-1} < w) - indicator(d_k < w)
    
    # Let's compute this efficiently
    active = (diffs < w[np.newaxis, :]).astype(float)  # shape (n-1, n_heights)
    
    # For each height, compute the gradient contribution per original index
    # grad[m, i] = sum over pairs involving i
    # 
    # Alternative: for each sorted position k, the net coefficient is:
    #   coeff[k] = indicator(d_{k-1} < w) - indicator(d_k < w)  (with d_{-1}=d_{n-1}=0)
    # Then grad_i = sum over k where sort_idx[k,m] == i of coeff[k,m]
    
    # coeff for sorted position k:
    # k=0: -active[0, m] (only d_0 matters, on the right side)
    # k=n-1: +active[n-2, m] (only d_{n-2} matters, on the left side)
    # k interior: +active[k-1, m] - active[k, m]
    
    coeff = np.zeros((n, n_heights))
    coeff[0, :] = -active[0, :]
    coeff[-1, :] = active[-2, :]
    for k in range(1, n-1):
        coeff[k, :] = active[k-1, :] - active[k, :]
    
    # Now scatter: grad[i, m] = coeff[k, m] where sort_idx[k, m] == i
    # Use np.add.at or bincount
    # For each height m, we need to scatter coeff[:, m] to positions given by sort_idx[:, m]
    
    # Vectorized approach using advanced indexing
    # Create a flat version
    flat_coeff = coeff.ravel()  # shape (n * n_heights)
    flat_idx = sort_idx.ravel()  # shape (n * n_heights), values in [0, n)
    
    # grad_flat[flat_idx] += flat_coeff
    grad_flat = np.zeros(n * n_heights)
    np.add.at(grad_flat, flat_idx, flat_coeff)
    
    # Average over heights to get the subgradient for the mean area
    grad = grad_flat.reshape(n_heights, n).mean(axis=0)
    
    return grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Direct construction for Kakeya needle triangle positions.
    
    Uses aligned initialization x_i = -a_i + 0.5, then subgradient polish
    and random local search to minimize the union area.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    a = (np.arange(n) + 1) / n
    n_heights = 20000
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    # Aligned initialization: x_i = -a_i + 0.5
    # At height y, L_i = -a_i + 0.5 + a_i * y = a_i * (y - 1) + 0.5
    # This centers the spread around 0.5 at all heights
    positions = -a + 0.5
    
    # Evaluate
    best_positions = positions.copy()
    best_area = _compute_area_fast(best_positions, a, y, w)
    
    # Try a few offset variations
    for offset in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if time.time() > deadline:
            break
        pos_try = -a + offset
        area_try = _compute_area_fast(pos_try, a, y, w)
        if area_try < best_area:
            best_area = area_try
            best_positions = pos_try.copy()
    
    # Subgradient descent polish
    if time.time() < deadline - 30:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.01
        
        np.random.seed(42)
        max_iter = 2000
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            grad = _subgradient(current, a, y, w, n, n_heights)
            
            # Line search: try a few step sizes
            best_trial_area = current_area
            best_trial_pos = current.copy()
            
            for lr_try in [lr, lr * 0.5, lr * 0.25]:
                trial = current - lr_try * grad
                trial_area = _compute_area_fast(trial, a, y, w)
                if trial_area < best_trial_area:
                    best_trial_area = trial_area
                    best_trial_pos = trial.copy()
            
            if best_trial_area < current_area:
                current = best_trial_pos
                current_area = best_trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
                lr = min(lr * 1.1, 0.05)
            else:
                lr *= 0.5
        
        # Random local search refinement
        if time.time() < deadline - 15:
            current = best_positions.copy()
            current_area = best_area
            step_size = 0.02
            
            np.random.seed(42)
            max_iter = 3000
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                idx = np.random.randint(n)
                delta = np.random.uniform(-step_size, step_size)
                trial = current.copy()
                trial[idx] += delta
                
                trial_area = _compute_area_fast(trial, a, y, w)
                
                if trial_area < current_area:
                    current = trial
                    current_area = trial_area
                    if current_area < best_area:
                        best_area = current_area
                        best_positions = current.copy()
                
                step_size = max(0.001, step_size * 0.9995)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        current = best_positions.copy()
        current_area = best_area
        
        # Subgradient with very small steps
        lr = 0.002
        np.random.seed(123)
        max_iter = 1000
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            grad = _subgradient(current, a, y, w, n, n_heights)
            
            best_trial_area = current_area
            best_trial_pos = current.copy()
            
            for lr_try in [lr, lr * 0.5]:
                trial = current - lr_try * grad
                trial_area = _compute_area_fast(trial, a, y, w)
                if trial_area < best_trial_area:
                    best_trial_area = trial_area
                    best_trial_pos = trial.copy()
            
            if best_trial_area < current_area:
                current = best_trial_pos
                current_area = best_trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            else:
                lr *= 0.5
        
        # Tiny random perturbations
        if time.time() < deadline - 2:
            current = best_positions.copy()
            current_area = best_area
            step_size = 0.002
            
            np.random.seed(999)
            max_iter = 2000
            for _ in range(max_iter):
                if time.time() > deadline - 1:
                    break
                
                idx = np.random.randint(n)
                delta = np.random.uniform(-step_size, step_size)
                trial = current.copy()
                trial[idx] += delta
                
                trial_area = _compute_area_fast(trial, a, y, w)
                
                if trial_area < current_area:
                    current = trial
                    current_area = trial_area
                    if current_area < best_area:
                        best_area = current_area
                        best_positions = current.copy()
                
                step_size = max(0.0005, step_size * 0.999)
    
    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
