def construct():
    import time
    import numpy as np
    from fractions import Fraction
    
    deadline = time.time() + 75
    
    denoms = list(range(2, 49))
    n = len(denoms)
    
    mid = n // 2
    left_vals = [1.0/d for d in denoms[:mid]]
    right_vals = [1.0/d for d in denoms[mid:]]
    
    n_left = len(left_vals)
    n_right = len(right_vals)
    
    # Generate left subset sums and masks
    left_sums = np.zeros(1 << n_left)
    left_masks = np.zeros(1 << n_left, dtype=np.int64)
    for i in range(n_left):
        v = left_vals[i]
        half = 1 << i
        left_sums[half:] = left_sums[:half] + v
        left_masks[half:] = left_masks[:half] | (1 << i)
    
    if time.time() > deadline - 30:
        return _greedy_fallback(denoms)
    
    # Generate right subset sums and masks
    right_sums = np.zeros(1 << n_right)
    right_masks = np.zeros(1 << n_right, dtype=np.int64)
    for i in range(n_right):
        v = right_vals[i]
        half = 1 << i
        right_sums[half:] = right_sums[:half] + v
        right_masks[half:] = right_masks[:half] | (1 << i)
    
    if time.time() > deadline - 20:
        return _greedy_fallback(denoms)
    
    # Sort right by sum
    order = np.argsort(right_sums)
    right_sums_sorted = right_sums[order]
    right_masks_sorted = right_masks[order]
    
    # Find best combination
    valid = left_sums < 1.0
    valid_idx = np.where(valid)[0]
    
    if len(valid_idx) == 0:
        return _greedy_fallback(denoms)
    
    targets = 1.0 - left_sums[valid_idx]
    idxs = np.searchsorted(right_sums_sorted, targets, side='left') - 1
    idxs = np.maximum(idxs, 0)
    
    best_sum = left_sums[valid_idx] + right_sums_sorted[idxs]
    best_local = np.argmax(best_sum)
    
    best_left_idx = valid_idx[best_local]
    best_right_idx = idxs[best_local]
    
    left_mask = left_masks[best_left_idx]
    right_mask = right_masks_sorted[best_right_idx]
    
    left_selected = [i for i in range(n_left) if left_mask & (1 << i)]
    right_selected = [mid + i for i in range(n_right) if right_mask & (1 << i)]
    
    all_selected = sorted(left_selected + right_selected)
    
    # Verify with exact fractions
    total = Fraction(0)
    for idx in all_selected:
        total += Fraction(1, denoms[idx])
    
    if total >= 1 or len(all_selected) == 0:
        return _greedy_fallback(denoms)
    
    return all_selected

def _greedy_fallback(denoms):
    from fractions import Fraction
    selected, total = [], Fraction()
    for i, d in enumerate(denoms):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
