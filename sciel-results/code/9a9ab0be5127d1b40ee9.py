# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(positions):
    """Compute the union area of the 64 triangles given base positions."""
    n = 64
    M = 20000
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    # L_i(y) = x_i + a_i * y, where a_i = (i+1)/64
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    # L shape: (M, n)
    L = positions[None, :] + a[None, :] * y[:, None]
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)
    w = (1.0 - y) / n  # width at each height, shape (M,)
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # (M, n-1)
    # min(diffs, w) where w is broadcast
    union_len = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    area = np.mean(union_len)
    return area


def _keich_positions():
    """Generate the Keich construction positions."""
    n = 64
    k = int(np.log2(n))
    positions = np.zeros(n, dtype=np.float64)
    j = np.arange(1, k + 1)
    for i in range(n):
        # Binary representation of i with k bits
        eps = np.zeros(k, dtype=np.float64)
        for bit in range(k):
            if i & (1 << bit):
                eps[bit] = 1.0
        # eps[j-1] corresponds to 2^{-j}
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    return positions


def _subgradient_descent(positions, max_iter, deadline, initial_lr=0.01):
    """Subgradient descent to minimize the area."""
    n = 64
    M = 20000
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n  # (M,)
    
    best_pos = positions.copy()
    best_area = _compute_area(best_pos)
    lr = initial_lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute L = x + a*y
        L = positions[None, :] + a[None, :] * y[:, None]  # (M, n)
        L_sorted = np.sort(L, axis=1)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # (M, n-1)
        
        # Union length gradient w.r.t. L
        # d(union_len)/dL_j for each element
        # For sorted L, the contribution of each gap min(d, w)
        # Subgradient: for each gap, if diff < w, it contributes 1 to the right element and -1 to the left
        # But we need to track which original index each sorted element came from
        
        # Actually, let's compute the subgradient more carefully.
        # union_len = w + sum_k min(L_sorted[k+1] - L_sorted[k], w)
        # d/dL_sorted[k] for k=0: -min(L_sorted[1]-L_sorted[0], w) * 1 (if diff<w) ... 
        # This is complex. Let me use a different approach.
        
        # Alternative: compute gradient via the indicator that each triangle's left edge
        # is on the boundary of the union.
        
        # Simpler: use finite differences or a direct subgradient.
        # For each triangle i, the subgradient of area w.r.t. x_i:
        # The triangle i contributes to the union at height y if its cross-section [L_i, L_i+w]
        # has parts not covered by other triangles.
        
        # Let me compute: for each height y_m, which triangles have their left endpoint
        # being the minimum or contributing to the union boundary.
        
        # Actually, the cleanest approach: 
        # union_len = w + sum_{k=0}^{n-2} min(L_sorted[k+1] - L_sorted[k], w)
        # The derivative w.r.t. L_sorted[k] (for the sorted array) is:
        #   -1 * I(L_sorted[k+1]-L_sorted[k] < w) for the left term of gap k
        #   +1 * I(L_sorted[k+1]-L_sorted[k] < w) for the right term of gap k
        # So for sorted position k:
        #   grad_sorted[k] = I(gap_{k-1} < w) - I(gap_k < w)  (with boundary terms)
        
        # Let me compute this properly.
        gap_lt_w = (diffs < w[:, None])  # (M, n-1) boolean
        
        # grad_sorted: (M, n)
        grad_sorted = np.zeros_like(L_sorted)
        # For each gap k (0 to n-2), if diff < w:
        #   grad_sorted[k] += 1 (as the right element of gap k-1... wait)
        # Let me re-derive:
        # union_len = w + sum_{k=0}^{n-2} min(L_s[k+1] - L_s[k], w)
        # d/dL_s[j] = sum_k d/dL_s[j] min(L_s[k+1]-L_s[k], w)
        # For a given gap k: d/dL_s[j] min(L_s[k+1]-L_s[k], w)
        #   if j == k: -1 * I(diff_k < w)
        #   if j == k+1: +1 * I(diff_k < w)
        #   else: 0
        # So: grad_sorted[j] = I(j>0 and gap_{j-1}<w) - I(j<n-1 and gap_j<w)
        
        # gap_k corresponds to L_s[k+1] - L_s[k], indexed 0 to n-2
        # For sorted index j:
        #   contribution from gap j-1 (if j>=1): +1 if gap_{j-1} < w
        #   contribution from gap j (if j<=n-2): -1 if gap_j < w
        
        if n > 1:
            # +1 from gap j-1 for j >= 1
            grad_sorted[:, 1:] += gap_lt_w
            # -1 from gap j for j <= n-2
            grad_sorted[:, :-1] -= gap_lt_w
        
        # Now map back to original indices
        # We need the permutation that sorts L at each height
        perm = np.argsort(L, axis=1)  # (M, n)
        
        # grad_original: (M, n)
        grad_original = np.zeros_like(L)
        # Scatter grad_sorted back
        # For each row m, grad_original[m, perm[m, j]] = grad_sorted[m, j]
        rows = np.arange(M)[:, None]
        grad_original[rows, perm] = grad_sorted
        
        # The area is mean over m of union_len[m]
        # d(area)/d(x_i) = (1/M) * sum_m grad_original[m, i]
        # Because L[m, i] = x_i + a_i * y_m, so dL/dx_i = 1
        grad = np.mean(grad_original, axis=0)
        
        # Update
        positions = positions - lr * grad
        area = _compute_area(positions)
        
        if area < best_area:
            best_area = area
            best_pos = positions.copy()
        
        # Decay learning rate
        lr *= 0.995
    
    return best_pos, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses Keich construction as starting point, then refines with subgradient descent.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Start with Keich construction
    pos = _keich_positions()
    best_pos = pos.copy()
    best_area = _compute_area(best_pos)
    
    # Phase 1: Coarse subgradient descent
    max_iter_phase1 = 2000
    pos, area = _subgradient_descent(pos, max_iter_phase1, deadline, initial_lr=0.05)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 10:
        return best_pos
    
    # Phase 2: Fine subgradient descent from best
    pos = best_pos.copy()
    max_iter_phase2 = 5000
    pos, area = _subgradient_descent(pos, max_iter_phase2, deadline, initial_lr=0.01)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 10:
        return best_pos
    
    # Phase 3: Even finer tuning
    pos = best_pos.copy()
    max_iter_phase3 = 10000
    pos, area = _subgradient_descent(pos, max_iter_phase3, deadline, initial_lr=0.003)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    return best_pos


# EVOLVE-BLOCK-END
