<<<<<<< SEARCH
import math
import time
from functools import reduce

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
    # Each index can be -1, 0, or +1
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
    
    # Convert B sums to sorted list for binary search
    arr_b = sorted(sums_b)
    b_min = arr_b[0]
    b_max = arr_b[-1]
    
    best_val = None
    best_a = None
    best_b = None
    
    # Sort A sums by absolute value to find small |a+b| faster
    a_list = sorted(sums_a, key=lambda a: abs(a))
    
    # Binary search helper
    import bisect
    
    for a in a_list:
        if time.time() > deadline:
            break
        
        # Pruning: if we already have a good answer and current a is too far
        # from being able to improve, we can stop
        # The best possible b for this a would be closest to -a
        # The minimum |a+b| achievable is at least the distance from -a to the nearest b
        if best_val is not None and best_val == 1:
            break
        
        target = -a
        if target < b_min:
            # Best b is b_min
            val = a + b_min
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_min
        elif target > b_max:
            # Best b is b_max
            val = a + b_max
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_max
        else:
            # target is within B range, use binary search
            idx = bisect.bisect_left(arr_b, target)
            
            # Check idx and idx-1
            for candidate_idx in [idx, idx - 1]:
                if 0 <= candidate_idx < len(arr_b):
                    b = arr_b[candidate_idx]
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
    
    # If we found a solution, reconstruct coefficients
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(best_a, 0, mid, W)
        coeffs_b = _find_coeffs(best_b, mid, n, W)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
    
    return _fallback(W, n)
=======
import math
import time
from functools import reduce
import numpy as np
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
    
    # Generate all possible sums for first half (indices 0..14) using numpy
    sums_a = _generate_sums_numpy(W[:mid])
    
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Generate all possible sums for second half (indices 15..29) using numpy
    sums_b = _generate_sums_numpy(W[mid:])
    
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Convert B sums to sorted numpy array for binary search
    arr_b = np.sort(sums_b)
    b_min = arr_b[0]
    b_max = arr_b[-1]
    
    best_val = None
    best_a = None
    best_b = None
    
    # Sort A sums by absolute value to find small |a+b| faster
    a_list = np.sort(sums_a, key=lambda a: abs(a))
    
    for a in a_list:
        if time.time() > deadline:
            break
        
        # Pruning: if we already have a good answer, we can stop early
        if best_val is not None and best_val == 1:
            break
        
        target = -a
        if target < b_min:
            # Best b is b_min
            val = a + b_min
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_min
        elif target > b_max:
            # Best b is b_max
            val = a + b_max
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_max
        else:
            # target is within B range, use binary search
            idx = bisect.bisect_left(arr_b, target)
            
            # Check idx and idx-1
            for candidate_idx in [idx, idx - 1]:
                if 0 <= candidate_idx < len(arr_b):
                    b = arr_b[candidate_idx]
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
    
    # If we found a solution, reconstruct coefficients
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(best_a, 0, mid, W)
        coeffs_b = _find_coeffs(best_b, mid, n, W)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
    
    return _fallback(W, n)

def _generate_sums_numpy(weights):
    """Generate all possible signed sums using numpy vectorization."""
    sums = np.array([0])
    for w in weights:
        # For each existing sum, create three new sums: s+w, s, s-w
        new_sums = np.concatenate([sums + w, sums, sums - w])
        sums = np.unique(new_sums)
    return sums
>>>>>>> REPLACE

