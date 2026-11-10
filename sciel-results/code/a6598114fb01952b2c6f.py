<<<<<<< SEARCH
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


def _compute_area_fast(positions, n=64, n_heights=20000):
    """Faster area computation using precomputed constants."""
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


def _subgradient(positions, n=64, n_heights=20000):
    """
    Compute subgradient of the union area with respect to positions.
    Vectorized implementation.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort and track original indices
    sort_idx = np.argsort(L, axis=0)  # (n, n_heights)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    
    # Mask where diff < w (overlapping)
    mask = diffs < w[np.newaxis, :]  # (n-1, n_heights)
    
    sg = np.zeros((n, n_heights))
    
    j_idx, m_idx = np.where(mask)
    
    if len(j_idx) > 0:
        left_tris = sort_idx[j_idx, m_idx]
        right_tris = sort_idx[j_idx + 1, m_idx]
        
        np.add.at(sg, (left_tris, m_idx), -1.0)
        np.add.at(sg, (right_tris, m_idx), 1.0)
        
    gradient = np.mean(sg, axis=1)
    
    return gradient


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Direct construction for Kakeya needle triangle positions.
    
    Uses the structural insight that triangles with larger a_i tilt more,
    so we offset their base positions to maximize overlap at all heights.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    a = (np.arange(n) + 1) / n
    
    # Try multiple initializations:
    # 1. Aligned: x_i = -a_i + 0.5 (near-optimal overlap by construction)
    # 2. Scaled: x_i = -a_i * c for various c
    # 3. Random perturbations of aligned
    
    candidates = []
    
    # Aligned construction
    candidates.append(-a + 0.5)
    
    # Various c values
    for c_try in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        candidates.append(-a * c_try + 0.5 * c_try)
    
    # Random perturbations of aligned
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        pert = np.random.normal(0, 0.05, n)
        candidates.append(-a + 0.5 + pert)
    
    # Evaluate all candidates
    best_positions = None
    best_area = np.inf
    
    for pos_try in candidates:
        if time.time() > deadline:
            break
        area_try = _compute_area(pos_try, n, 20000)
        if area_try < best_area:
            best_area = area_try
            best_positions = pos_try.copy()
    
    # Local search phase 1: coarse
    if time.time() < deadline - 60:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.03
        
        np.random.seed(42)
        for it in range(8000):
            if time.time() > deadline - 60:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.005, step_size * 0.9997)
    
    # Local search phase 2: fine
    if time.time() < deadline - 30:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.01
        
        np.random.seed(123)
        for it in range(6000):
            if time.time() > deadline - 30:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.001, step_size * 0.9995)
    
    # Subgradient polish phase
    if time.time() < deadline - 10:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.05
        
        for it in range(20000):
            if time.time() > deadline - 10:
                break
            
            grad = _subgradient(current, n, 20000)
            current = current - lr * grad
            current_area = _compute_area(current, n, 20000)
            
            if current_area < best_area:
                best_area = current_area
                best_positions = current.copy()
            
            lr *= 0.9995
    
    # Final micro-refinement
    if time.time() < deadline - 3:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.002
        
        np.random.seed(999)
        for it in range(2000):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
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
=======
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


def _compute_area_grad(positions, n=64, n_heights=20000):
    """Compute area and subgradient together to avoid redundant work."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    mask = diffs < w[np.newaxis, :]
    sg = np.zeros((n, n_heights))
    
    j_idx, m_idx = np.where(mask)
    if len(j_idx) > 0:
        left_tris = sort_idx[j_idx, m_idx]
        right_tris = sort_idx[j_idx + 1, m_idx]
        np.add.at(sg, (left_tris, m_idx), -1.0)
        np.add.at(sg, (right_tris, m_idx), 1.0)
    
    gradient = np.mean(sg, axis=1)
    return area, gradient


def _soft_min(x, softness=0.01):
    """Soft-min approximation via log-sum-exp for smoothing."""
    # soft_min(x) ≈ -softness * log(sum(exp(-x/softness)))
    # But we want min(diffs, w), so use:
    # soft_min(d, w) = (d - w - softness * log(1 + exp((d-w)/softness))) / 2 + ...
    # Simpler: use the fact that min(a,b) = (a+b-|a-b|)/2
    # and smooth |x| with soft_abs
    return x  # placeholder, we'll use direct approach below


def _line_search_coord(positions, idx, n=64, n_heights=20000, max_iter=50):
    """
    1D line search along coordinate idx using binary search on the
    piecewise-linear function. Returns optimal value for positions[idx].
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    # Precompute L for all other triangles (fixed)
    other_idx = [i for i in range(n) if i != idx]
    L_other = positions[other_idx, np.newaxis] + a[other_idx, np.newaxis] * y[np.newaxis, :]
    L_other_sorted = np.sort(L_other, axis=0)
    
    # Current value
    x_cur = positions[idx]
    
    def area_at_x(x_val):
        """Compute area when positions[idx] = x_val."""
        L_idx = x_val + a[idx] * y
        # Merge L_idx into sorted L_other
        # For each height, insert L_idx into L_other_sorted
        # Use searchsorted to find insertion point
        insert_pos = np.searchsorted(L_other_sorted, L_idx, side='left')
        
        # Build full sorted array
        L_full = np.empty((n, n_heights))
        L_full[0, :] = L_other_sorted[0, :]
        for k in range(1, n - 1):
            L_full[k, :] = L_other_sorted[k, :]
        
        # Actually, let's do this more efficiently:
        # For each height, we have n-1 sorted values and need to insert one more.
        # The union length = w + sum min(diff_j, w) for j=0..n-2
        # When we insert L_idx, it splits one gap into two (or extends an end).
        
        # More efficient: compute the change in union length
        # Find where L_idx falls in the sorted order
        # The gap it splits is between L_sorted[insert_pos-1] and L_sorted[insert_pos]
        # (with appropriate boundary handling)
        
        # Let's just compute directly for correctness
        L_all = np.empty((n, n_heights))
        L_all[idx, :] = L_idx
        
        # For each height, merge
        # Use the fact that we can compute union length from sorted values
        # We'll compute L_all sorted for each height
        
        # Efficient approach: for each height, the sorted order of other triangles is fixed.
        # We just need to insert L_idx.
        # Union length = w + sum_{j=0}^{n-2} min(d_j, w) where d_j are consecutive diffs.
        
        # When inserting L_idx at position p (0-indexed) in the sorted (n-1) array:
        # - If p == 0: new gap is L_idx - L_sorted[0], and L_sorted[0] to L_sorted[1] remains
        #   Actually the first gap changes from (L_sorted[0] to L_sorted[1])... no.
        #   The original gaps are: (L_sorted[0], L_sorted[1]), ..., (L_sorted[n-3], L_sorted[n-2])
        #   Wait, L_other_sorted has shape (n-1, n_heights), so diffs are (n-2, n_heights).
        #   Original union = w + sum_{j=0}^{n-3} min(L_other_sorted[j+1]-L_other_sorted[j], w)
        #   
        #   After inserting L_idx at position p:
        #   If p == 0: new first element is L_idx, so new gap0 = L_other_sorted[0] - L_idx
        #              gaps 1..n-3 unchanged, but we now have n-1 gaps total.
        #              New union = w + min(L_other_sorted[0]-L_idx, w) + sum_{j=1}^{n-3} min(L_other_sorted[j+1]-L_other_sorted[j], w)
        #   If p == n-1: L_idx is largest, new last gap = L_idx - L_other_sorted[n-2]
        #   If 0 < p < n-1: gap p-1 was L_other_sorted[p]-L_other_sorted[p-1], now split into
        #                    (L_idx - L_other_sorted[p-1]) and (L_other_sorted[p] - L_idx)
        
        # Let's compute the original sum of min diffs
        orig_diffs = np.diff(L_other_sorted, axis=0)  # (n-2, n_heights)
        orig_overlaps = np.minimum(orig_diffs, w[np.newaxis, :])
        orig_sum = np.sum(orig_overlaps, axis=0)  # (n_heights,)
        
        # For each height, compute new sum
        # insert_pos is (n_heights,) array
        new_sum = np.zeros(n_heights)
        
        # Case 1: insert at position 0
        mask0 = (insert_pos == 0)
        if np.any(mask0):
            new_gap = L_other_sorted[0, mask0] - L_idx[mask0]
            new_sum[mask0] = np.minimum(new_gap, w[mask0]) + orig_sum[mask0]
        
        # Case 2: insert at position n-1 (end)
        mask_end = (insert_pos == n - 1)
        if np.any(mask_end):
            new_gap = L_idx[mask_end] - L_other_sorted[n - 2, mask_end]
            new_sum[mask_end] = np.minimum(new_gap, w[mask_end]) + orig_sum[mask_end]
        
        # Case 3: insert in middle
        mask_mid = (insert_pos > 0) & (insert_pos < n - 1)
        if np.any(mask_mid):
            p = insert_pos[mask_mid]
            # The gap being split is between L_other_sorted[p-1] and L_other_sorted[p]
            old_gap = L_other_sorted[p, :] - L_other_sorted[p - 1, :]
            old_min = np.minimum(old_gap, w)
            
            new_gap1 = L_idx[mask_mid] - L_other_sorted[p - 1, :]
            new_gap2 = L_other_sorted[p, :] - L_idx[mask_mid]
            new_min_sum = np.minimum(new_gap1, w[mask_mid]) + np.minimum(new_gap2, w[mask_mid])
            
            new_sum[mask_mid] = orig_sum[mask_mid] - old_min[mask_mid] + new_min_sum[mask_mid]
        
        union_lengths = w + new_sum
        return np.mean(union_lengths)
    
    # Binary search for minimum in a reasonable range around current value
    x_lo = x_cur - 2.0
    x_hi = x_cur + 2.0
    
    # First, find a bracket
    f_lo = area_at_x(x_lo)
    f_mid = area_at_x(x_cur)
    f_hi = area_at_x(x_hi)
    
    # Expand if needed
    for _ in range(10):
        if f_lo > f_mid:
            x_lo -= 1.0
            f_lo = area_at_x(x_lo)
        elif f_hi > f_mid:
            x_hi += 1.0
            f_hi = area_at_x(x_hi)
        else:
            break
    
    # Now binary search
    best_x = x_cur
    best_f = f_mid
    
    for _ in range(max_iter):
        x1 = x_lo + (x_hi - x_lo) / 3
        x2 = x_hi - (x_hi - x_lo) / 3
        f1 = area_at_x(x1)
        f2 = area_at_x(x2)
        
        if f1 < f2:
            x_hi = x2
            if f1 < best_f:
                best_f = f1
                best_x = x1
        else:
            x_lo = x1
            if f2 < best_f:
                best_f = f2
                best_x = x2
        
        if abs(x_hi - x_lo) < 1e-7:
            break
    
    return best_x, best_f


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Direct construction for Kakeya needle triangle positions.
    
    Uses coordinate descent with line search for efficient 1D optimization,
    combined with subgradient descent for global structure.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    a = (np.arange(n) + 1) / n
    
    # Try multiple initializations
    candidates = []
    
    # Aligned construction
    candidates.append(-a + 0.5)
    
    # Various c values
    for c_try in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        candidates.append(-a * c_try + 0.5 * c_try)
    
    # Random perturbations of aligned
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        pert = np.random.normal(0, 0.05, n)
        candidates.append(-a + 0.5 + pert)
    
    # Evaluate all candidates
    best_positions = None
    best_area = np.inf
    
    for pos_try in candidates:
        if time.time() > deadline:
            break
        area_try = _compute_area(pos_try, n, 20000)
        if area_try < best_area:
            best_area = area_try
            best_positions = pos_try.copy()
    
    # Coordinate descent phase with line search
    if time.time() < deadline - 60:
        current = best_positions.copy()
        current_area = best_area
        np.random.seed(42)
        
        # Cycle through coordinates multiple times
        max_cd_iters = 5000
        for it in range(max_cd_iters):
            if time.time() > deadline - 60:
                break
            
            # Random coordinate order for diversity
            idx = np.random.randint(n)
            
            new_x, new_f = _line_search_coord(current, idx, n, 20000, max_iter=30)
            
            if new_f < current_area - 1e-12:
                current[idx] = new_x
                current_area = new_f
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
    
    # Subgradient phase with combined area+grad computation
    if time.time() < deadline - 30:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.05
        
        for it in range(15000):
            if time.time() > deadline - 30:
                break
            
            area, grad = _compute_area_grad(current, n, 20000)
            current = current - lr * grad
            current_area = _compute_area(current, n, 20000)
            
            if current_area < best_area:
                best_area = current_area
                best_positions = current.copy()
            
            lr *= 0.9995
    
    # Final coordinate descent polish
    if time.time() < deadline - 5:
        current = best_positions.copy()
        current_area = best_area
        np.random.seed(999)
        
        for it in range(3000):
            if time.time() > deadline - 5:
                break
            
            idx = np.random.randint(n)
            new_x, new_f = _line_search_coord(current, idx, n, 20000, max_iter=40)
            
            if new_f < current_area - 1e-12:
                current[idx] = new_x
                current_area = new_f
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
    
    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
>>>>>>> REPLACE
