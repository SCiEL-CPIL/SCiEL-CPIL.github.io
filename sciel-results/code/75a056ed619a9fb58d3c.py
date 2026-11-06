import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = 15
    
    # Generate all possible sums for first half (indices 0..14)
    sums_a = {0}
    for i in range(mid):
        w = W[i]
        new_sums = set()
        for s in sums_a:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_a = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Generate all possible sums for second half (indices 15..29)
    sums_b = {0}
    for i in range(mid, n):
        w = W[i]
        new_sums = set()
        for s in sums_b:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_b = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Convert to sorted numpy arrays
    arr_a = np.array(list(sums_a), dtype=np.int64)
    arr_b = np.array(sorted(sums_b), dtype=np.int64)
    
    # Vectorized search
    # We want to minimize |a + b| != 0
    # For each a, find b closest to -a
    # idx = searchsorted(arr_b, -arr_a)
    # Candidates are arr_b[idx] and arr_b[idx-1]
    
    neg_a = -arr_a
    idx = np.searchsorted(arr_b, neg_a)
    
    # Handle boundaries
    # Left candidates: idx - 1 (if valid)
    # Right candidates: idx (if valid)
    
    # Create arrays for left and right candidates
    # We need to be careful with indices out of bounds
    
    # Right candidates
    idx_right = np.clip(idx, 0, len(arr_b) - 1)
    b_right = arr_b[idx_right]
    
    # Left candidates
    idx_left = np.clip(idx - 1, 0, len(arr_b) - 1)
    b_left = arr_b[idx_left]
    
    # Calculate sums
    sum_right = arr_a + b_right
    sum_left = arr_a + b_left
    
    # We want the minimum non-zero absolute value
    # Mask out zeros
    mask_right = sum_right != 0
    mask_left = sum_left != 0
    
    # Find min absolute values
    abs_right = np.abs(sum_right).astype(np.int64)
    abs_left = np.abs(sum_left).astype(np.int64)
    
    # Set invalid ones to a large sentinel value
    BIG = 10**18
    abs_right[~mask_right] = BIG
    abs_left[~mask_left] = BIG
    
    # Find the global minimum
    min_abs_right = np.min(abs_right)
    min_abs_left = np.min(abs_left)
    
    if min_abs_right < min_abs_left:
        best_val = min_abs_right
        best_idx = np.argmin(abs_right)
        best_a = arr_a[best_idx]
        best_b = b_right[best_idx]
    else:
        best_val = min_abs_left
        best_idx = np.argmin(abs_left)
        best_a = arr_a[best_idx]
        best_b = b_left[best_idx]
        
    if best_val >= BIG:
        return _fallback(W, n)
        
    # Reconstruct coefficients
    coeffs_a = _find_coeffs(int(best_a), 0, mid, W)
    coeffs_b = _find_coeffs(int(best_b), mid, n, W)
    
    if coeffs_a is not None and coeffs_b is not None:
        final_coeffs = coeffs_a + coeffs_b
        total = sum(c * W[i] for i, c in enumerate(final_coeffs))
        if total != 0:
            return final_coeffs
    
    return _fallback(W, n)

def _find_coeffs(target, start, end, W):
    """Recursively find coefficients for indices start to end (0-based, exclusive end)."""
    if start == end:
        return [] if target == 0 else None
        
    # Pruning: max possible absolute sum from start to end
    max_rem = sum(W[start:end])
    if abs(target) > max_rem:
        return None
        
    w = W[start]
    for c in [-1, 0, 1]:
        res = _find_coeffs(target - c * w, start + 1, end, W)
        if res is not None:
            return [c] + res
    return None

def _fallback(W, n):
    """Greedy fallback: construct a small nonzero sum."""
    coeffs = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        w = W[i]
        best_c = 0
        best_val = curr
        for c in [-1, 0, 1]:
            val = curr + c * w
            if abs(val) < abs(best_val):
                best_val = val
                best_c = c
        coeffs[i] = best_c
        curr = best_val
    
    if curr == 0:
        # Make nonzero by setting smallest weight to 1
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs[min_idx] = 1
        curr = sum(c * W[i] for i, c in enumerate(coeffs))
    
    return coeffs
