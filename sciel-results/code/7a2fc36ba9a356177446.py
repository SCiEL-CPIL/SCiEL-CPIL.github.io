import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = 30
    L = reduce(math.lcm, range(1, n + 1))
    W = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = 15
    W_a = W[:mid]
    W_b = W[mid:]
    
    # Generate all possible sums for first half using NumPy
    # Start with [0]
    sums_a = np.array([0], dtype=np.int64)
    for w in W_a:
        # Create new array with 3x size
        # sums_a, sums_a + w, sums_a - w
        sums_a = np.concatenate((sums_a, sums_a + w, sums_a - w))
        
    if time.time() > deadline:
        return _fallback(W, n)
        
    # Generate all possible sums for second half using NumPy
    sums_b = np.array([0], dtype=np.int64)
    for w in W_b:
        sums_b = np.concatenate((sums_b, sums_b + w, sums_b - w))
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Convert sums_b to a set for O(1) lookup
    # This might take some time/memory but is necessary for fast target search
    set_b = set(sums_b.tolist())
    
    # We want to find the smallest non-zero |a + b|
    # Check targets in order of increasing absolute value: 1, -1, 2, -2, ...
    # If we find a target T, we can reconstruct the solution
    
    found_target = None
    found_a = None
    found_b = None
    
    # Check small targets first
    for t_abs in range(1, 100):
        for t in [t_abs, -t_abs]:
            if time.time() > deadline:
                break
                
            # Check if there exists a in sums_a such that t - a is in set_b
            # This is equivalent to a + b = t
            # We can iterate sums_a and check (t - a) in set_b
            
            # Optimization: If t is small, and sums are large, 
            # we might not find it. But we must check.
            
            # To speed up, we can vectorize the check?
            # targets_in_b = t - sums_a
            # Then check which of these are in set_b.
            # This creates an array of size 1.4M.
            
            targets_in_b = t - sums_a
            
            # Find indices where targets_in_b is in set_b
            # We can use a list comprehension, but it's slow.
            # Better: convert set_b to a numpy array? No, lookup is slow.
            
            # Let's just loop. 1.4M iterations in Python is ~0.5-1s.
            # We are doing this for t=1, -1, 2, -2...
            # If 1 is the answer, we stop after first iteration.
            
            for a in sums_a:
                b = t - a
                if b in set_b:
                    found_target = t
                    found_a = a
                    found_b = b
                    break
            if found_target is not None:
                break
        if found_target is not None:
            break
        if time.time() > deadline:
            break
            
    if found_target is not None:
        coeffs_a = _find_coeffs(found_a, W_a)
        coeffs_b = _find_coeffs(found_b, W_b)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total == found_target:
                return final_coeffs
            # If reconstruction failed or sum mismatch, fallback
            return _fallback(W, n)
            
    # If small targets not found, find the true minimum using sorted array
    # This is slower, so we only do it if necessary
    arr_b_sorted = np.sort(sums_b)
    
    best_val = None
    best_a = None
    best_b = None
    
    # Iterate through sums_a
    # We can prune if we have found a very small value, but we don't know the min yet.
    # Just iterate.
    for a in sums_a:
        if time.time() > deadline:
            break
            
        # Find closest b to -a
        target = -a
        idx = np.searchsorted(arr_b_sorted, target)
        
        candidates = []
        if idx < len(arr_b_sorted):
            candidates.append(arr_b_sorted[idx])
        if idx > 0:
            candidates.append(arr_b_sorted[idx - 1])
            
        for b in candidates:
            val = a + b
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b
                if best_val == 1:
                    break
        if best_val == 1:
            break
            
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(best_a, W_a)
        coeffs_b = _find_coeffs(best_b, W_b)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
                
    return _fallback(W, n)

def _find_coeffs(target, W):
    """Recursively find coefficients for the given weights to sum to target."""
    if not W:
        return [] if target == 0 else None
        
    w = W[0]
    rest_W = W[1:]
    
    for c in [-1, 0, 1]:
        res = _find_coeffs(target - c * w, rest_W)
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
