import numpy as np
from scipy.optimize import minimize
import time

def _compute_area(positions, a, y, w, initial_perm=None):
    """Compute the union area of the 64 triangles given base positions."""
    L = positions[None, :] + a[None, :] * y[:, None]
    if initial_perm is None:
        perm = np.argsort(L, axis=1)
    else:
        perm = initial_perm
    L_sorted = np.take_along_axis(L, perm, axis=1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    return np.mean(union_lengths)

def _keich_positions(n=64):
    """Generate the Keich construction positions."""
    k = int(np.log2(n))
    positions = np.zeros(n, dtype=np.float64)
    j = np.arange(1, k + 1, dtype=np.float64)
    for i in range(n):
        eps = np.zeros(k, dtype=np.float64)
        for bit in range(k):
            if i & (1 << bit):
                eps[bit] = 1.0
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    return positions

def _subgradient_descent(positions, a, y, w, max_iter, deadline, initial_lr=0.01):
    """Subgradient descent to minimize the area with adaptive step size."""
    n = len(positions)
    M = len(y)
    
    best_pos = positions.copy()
    best_area = _compute_area(best_pos, a, y, w)
    lr = initial_lr
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute L = x + a*y
        L = positions[None, :] + a[None, :] * y[:, None]  # (M, n)
        L_sorted = np.sort(L, axis=1)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # (M, n-1)
        
        # Subgradient computation
        gap_lt_w = (diffs < w[:, None])  # (M, n-1) boolean
        
        # grad_sorted: (M, n)
        grad_sorted = np.zeros_like(L_sorted)
        
        if n > 1:
            # +1 from gap j-1 for j >= 1
            grad_sorted[:, 1:] += gap_lt_w
            # -1 from gap j for j <= n-2
            grad_sorted[:, :-1] -= gap_lt_w
        
        # Map back to original indices
        perm = np.argsort(L, axis=1)  # (M, n)
        
        # grad_original: (M, n)
        grad_original = np.zeros_like(L)
        rows = np.arange(M)[:, None]
        grad_original[rows, perm] = grad_sorted
        
        # The area is mean over m of union_len[m]
        # d(area)/d(x_i) = (1/M) * sum_m grad_original[m, i]
        grad = np.mean(grad_original, axis=0)
        
        # Update
        positions = positions - lr * grad
        area = _compute_area(positions, a, y, w)
        
        if area < best_area:
            best_area = area
            best_pos = positions.copy()
            no_improve_count = 0
            # Increase lr slightly if we just improved
            lr *= 1.02
        else:
            no_improve_count += 1
            # Decrease lr if no improvement
            lr *= 0.95
        
        # Keep lr in reasonable bounds
        lr = max(lr, 0.001)
        lr = min(lr, 0.5)
    
    return best_pos, best_area

def _coordinate_descent(positions, a, y, w, deadline, step_sizes=None):
    """Coordinate descent fine-tuning."""
    if step_sizes is None:
        step_sizes = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    
    n = len(positions)
    best_pos = positions.copy()
    best_area = _compute_area(best_pos, a, y, w)
    
    for step in step_sizes:
        if time.time() > deadline - 10:
            break
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 10:
                break
            improved = False
            for idx in range(n):
                if time.time() > deadline - 10:
                    break
                # Try positive and negative steps
                for sign in [1, -1]:
                    x_new = best_pos.copy()
                    x_new[idx] += sign * step
                    area_new = _compute_area(x_new, a, y, w)
                    if area_new < best_area:
                        best_pos = x_new
                        best_area = area_new
                        improved = True
            if not improved:
                break
    
    return best_pos, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the 64 base positions x_i to minimize the area of the union
    of the corresponding triangles. Combines multi-start Keich variants,
    L-BFGS-B, adaptive subgradient descent, and coordinate descent.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    # Precompute fixed values
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    M = 20000
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    
    # Multi-start: try several shuffled Keich variants
    rng = np.random.RandomState(42)
    best_pos = None
    best_area = np.inf
    
    # Original Keich
    pos0 = _keich_positions(n)
    area0 = _compute_area(pos0, a, y, w)
    best_pos = pos0.copy()
    best_area = area0
    
    # Shuffled variants
    for trial in range(5):
        if time.time() > deadline - 500:
            break
        perm = rng.permutation(n)
        pos_shuffled = pos0[perm]
        area_s = _compute_area(pos_shuffled, a, y, w)
        if area_s < best_area:
            best_area = area_s
            best_pos = pos_shuffled.copy()
    
    # Also try random perturbations of Keich
    for trial in range(3):
        if time.time() > deadline - 500:
            break
        pos_perturbed = pos0 + rng.normal(0, 0.05, n)
        area_p = _compute_area(pos_perturbed, a, y, w)
        if area_p < best_area:
            best_area = area_p
            best_pos = pos_perturbed.copy()
    
    # Phase 1: L-BFGS-B optimization from best start
    if time.time() < deadline - 150:
        try:
            res = minimize(
                lambda x: _compute_area(x, a, y, w),
                best_pos,
                method='L-BFGS-B',
                jac='2-point',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 20
                }
            )
            if res.fun < best_area:
                best_area = res.fun
                best_pos = res.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 100:
        return best_pos
    
    # Phase 2: Coarse adaptive subgradient descent
    if time.time() < deadline - 60:
        pos, area = _subgradient_descent(best_pos, a, y, w, 1500, deadline - 60, initial_lr=0.05)
        if area < best_area:
            best_area = area
            best_pos = pos.copy()
    
    if time.time() > deadline - 40:
        return best_pos
    
    # Phase 3: Fine adaptive subgradient descent
    if time.time() < deadline - 25:
        pos, area = _subgradient_descent(best_pos, a, y, w, 2000, deadline - 25, initial_lr=0.01)
        if area < best_area:
            best_area = area
            best_pos = pos.copy()
    
    if time.time() > deadline - 15:
        return best_pos
    
    # Phase 4: Coordinate descent fine-tuning
    if time.time() < deadline - 10:
        best_pos, best_area = _coordinate_descent(best_pos, a, y, w, deadline - 5)
    
    return best_pos
