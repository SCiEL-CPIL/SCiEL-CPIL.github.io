# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area_fast(x, n=64, M=20000):
    """
    Fast quadrature-based area computation (matches the evaluator exactly).
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def _compute_area_and_subgrad(x, n=64, M=20000):
    """
    Compute area and exact subgradient using the piecewise-linear structure.
    
    The area = mean over y of [w(y) + sum_j min(gap_j(y), w(y))]
    where gap_j(y) = L_{(j+1)}(y) - L_{(j)}(y) for sorted lines.
    
    For each y, the subgradient w.r.t. x_i is:
    - If line i is the leftmost in sorted order: contributes -1 to each gap it's part of
    - More precisely, track which line is at each position in sorted order
    
    Actually, let's use a cleaner approach:
    area = integral of union_length(y) dy
    union_length(y) = w(y) + sum_{j=1}^{n-1} min(gap_j(y), w(y))
    
    For a fixed y, if we know the sorted order, gap_j(y) = L_{s_{j+1}}(y) - L_{s_j}(y)
    where s is the sorted permutation.
    
    min(gap_j, w) = gap_j if gap_j < w, else w
    
    d/dx_i of gap_j = a_{s_{j+1}}*delta_{i,s_{j+1}} - a_{s_j}*delta_{i,s_j}
    
    So subgrad_i = sum over y of [sum over j where gap_j < w: (a_{s_{j+1}}*delta_{i,s_{j+1}} - a_{s_j}*delta_{i,s_j})]
    
    This can be computed by accumulating contributions.
    """
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    # L[i, m] = x_i + a_i * y_m
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort and track indices
    L_sorted = np.sort(L, axis=0)
    idx_sorted = np.argsort(L, axis=0)  # idx_sorted[m, j] = index of j-th smallest line at height m
    
    d = L_sorted[1:, :] - L_sorted[:-1, :]  # d[m, j] = gap between j-th and (j+1)-th line
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    
    # Compute subgradient
    # For each height m, for each gap j where d[m,j] < w[m]:
    #   subgrad[idx_sorted[m, j+1]] += a[idx_sorted[m, j+1]]
    #   subgrad[idx_sorted[m, j]] -= a[idx_sorted[m, j]]
    # Then multiply by 1/M (since we're averaging over M heights)
    
    subgrad = np.zeros(n)
    # active[m, j] = True if d[m,j] < w[m]
    active = d < w[np.newaxis, :]
    
    # For each active gap, add contributions
    # Vectorized: for each column j (gap index), find active rows
    for j in range(n - 1):
        active_mask = active[:, j]  # M booleans
        if not np.any(active_mask):
            continue
        # Lines involved in this gap at active heights
        left_lines = idx_sorted[active_mask, j]      # M indices
        right_lines = idx_sorted[active_mask, j + 1]  # M indices
        # Accumulate: subgrad[right] += a[right], subgrad[left] -= a[left]
        np.add.at(subgrad, right_lines, a[right_lines])
        np.add.at(subgrad, left_lines, -a[left_lines])
    
    subgrad /= M  # Because area is mean over M heights
    
    return float(area), subgrad

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill
    
    # Initial guess: try several configurations
    a = (np.arange(n) + 1) / n
    
    inits = [
        np.zeros(n),
        -a * 0.5,
        -a * (1.0/3.0),
        -a * (2.0/3.0),
        a * 0.5,
        -a * 0.25,
        -a * 0.75,
        -a * 0.4,
        -a * 0.6,
    ]
    
    best_x = inits[0].copy()
    best_area = _compute_area_fast(inits[0], n, M)
    
    for init in inits[1:]:
        area_init = _compute_area_fast(init, n, M)
        if area_init < best_area:
            best_area = area_init
            best_x = init.copy()
    
    # Subgradient descent with backtracking line search
    x = best_x.copy()
    current_area = best_area
    
    # Learning rate schedule
    lr_init = 0.1
    lr_min = 1e-4
    
    max_iters = 2000
    for iteration in range(max_iters):
        if time.time() > deadline - 30:
            break
        
        # Compute area and subgradient
        area, grad = _compute_area_and_subgrad(x, n, M)
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
        
        # Backtracking line search
        lr = lr_init * (0.95 ** (iteration // 50))
        if lr < lr_min:
            lr = lr_min
        
        # Try the step
        x_new = x - lr * grad
        area_new = _compute_area_fast(x_new, n, M)
        
        # Backtrack if needed
        backtracks = 0
        while area_new > area - 1e-10 * lr * grad_norm and backtracks < 10:
            lr *= 0.5
            x_new = x - lr * grad
            area_new = _compute_area_fast(x_new, n, M)
            backtracks += 1
        
        if area_new < area:
            x = x_new
            current_area = area_new
        else:
            # If we couldn't improve, reduce learning rate further
            lr_init *= 0.8
            if lr_init < lr_min:
                break
    
    # Final polish with coordinate descent for fine-tuning
    if time.time() < deadline - 60:
        x = best_x.copy()
        current_area = best_area
        
        for coord_iter in range(50):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                # Try small perturbations
                best_local = current_area
                best_delta = 0.0
                for delta in [-0.01, -0.005, -0.002, -0.001, 0.001, 0.002, 0.005, 0.01]:
                    x_trial = x.copy()
                    x_trial[i] += delta
                    area_trial = _compute_area_fast(x_trial, n, M)
                    if area_trial < best_local - 1e-12:
                        best_local = area_trial
                        best_delta = delta
                
                if best_delta != 0.0:
                    x[i] += best_delta
                    current_area = best_local
                    improved = True
            
            if not improved:
                # Try even smaller steps
                improved_small = False
                for i in range(n):
                    if time.time() > deadline - 30:
                        break
                    for delta in [-0.0005, -0.0002, -0.0001, 0.0001, 0.0002, 0.0005]:
                        x_trial = x.copy()
                        x_trial[i] += delta
                        area_trial = _compute_area_fast(x_trial, n, M)
                        if area_trial < current_area - 1e-12:
                            x[i] += delta
                            current_area = area_trial
                            improved_small = True
                            break
                    if improved_small:
                        break
                if not improved_small:
                    break
            
            if current_area < best_area:
                best_area = current_area
                best_x = x.copy()
    
    # Validation
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    if best_area <= 0:
        return np.zeros(n, dtype=np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
