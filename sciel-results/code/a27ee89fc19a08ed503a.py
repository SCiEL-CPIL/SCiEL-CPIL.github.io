import math
import time
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 30
    # Compute LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = 15
    
    # Generate all possible sums for first half (indices 0..14)
    sums_a = [0]
    for i in range(mid):
        w = W[i]
        new_sums = []
        append = new_sums.append
        for s in sums_a:
            append(s + w)
            append(s)
            append(s - w)
        sums_a = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
        
    # Sort and unique A
    sums_a.sort()
    unique_a = []
    if sums_a:
        last = None
        for s in sums_a:
            if s != last:
                unique_a.append(s)
                last = s
    sums_a = unique_a

    # Generate all possible sums for second half (indices 15..29)
    sums_b = [0]
    for i in range(mid, n):
        w = W[i]
        new_sums = []
        append = new_sums.append
        for s in sums_b:
            append(s + w)
            append(s)
            append(s - w)
        sums_b = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)

    # Sort and unique B
    sums_b.sort()
    unique_b = []
    if sums_b:
        last = None
        for s in sums_b:
            if s != last:
                unique_b.append(s)
                last = s
    sums_b = unique_b

    best_val = None
    best_a = None
    best_b = None
    
    # Two pointers
    i = 0
    j = len(sums_b) - 1
    n_a = len(sums_a)
    n_b = len(sums_b)
    
    while i < n_a and j >= 0:
        if time.time() > deadline:
            break
            
        s = sums_a[i] + sums_b[j]
        
        if s == 0:
            # Check immediate neighbors for non-zero minimum
            # Candidates: (i+1, j) and (i, j-1)
            # Also (i+1, j-1) might be closer?
            # Let's check a small window
            for di in [0, 1, -1]:
                for dj in [0, 1, -1]:
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < n_a and 0 <= nj < n_b:
                        val = sums_a[ni] + sums_b[nj]
                        if val != 0:
                            if best_val is None or abs(val) < best_val:
                                best_val = abs(val)
                                best_a = sums_a[ni]
                                best_b = sums_b[nj]
            if best_val == 1:
                break
            # Move pointers to explore other 0s or move away
            i += 1
            j -= 1
        elif s > 0:
            # s is positive, we want to decrease it
            # Current s is a candidate
            if best_val is None or s < best_val:
                best_val = s
                best_a = sums_a[i]
                best_b = sums_b[j]
                if best_val == 1:
                    break
            j -= 1
        else:
            # s is negative, we want to increase it
            # Current s is a candidate
            if best_val is None or -s < best_val:
                best_val = -s
                best_a = sums_a[i]
                best_b = sums_b[j]
                if best_val == 1:
                    break
            i += 1
            
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(best_a, 0, mid, W)
        coeffs_b = _find_coeffs(best_b, mid, n, W)
        
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
