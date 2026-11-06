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
    
    # Split into three groups of 10
    g1 = 10
    g2 = 10
    g3 = 10
    
    # Generate all possible sums with coefficient arrays for group 1 (indices 0..9)
    sums_a = {0: [0] * g1}
    for i in range(g1):
        w = W[i]
        new_sums = {}
        for s, coeffs in sums_a.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                if ns not in new_sums:
                    new_coeffs = list(coeffs)
                    new_coeffs[i] = c
                    new_sums[ns] = new_coeffs
        sums_a = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Generate all possible sums with coefficient arrays for group 2 (indices 10..19)
    sums_b = {0: [0] * g2}
    for i in range(g2):
        w = W[10 + i]
        new_sums = {}
        for s, coeffs in sums_b.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                if ns not in new_sums:
                    new_coeffs = list(coeffs)
                    new_coeffs[i] = c
                    new_sums[ns] = new_coeffs
        sums_b = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Generate all possible sums with coefficient arrays for group 3 (indices 20..29)
    sums_c = {0: [0] * g3}
    for i in range(g3):
        w = W[20 + i]
        new_sums = {}
        for s, coeffs in sums_c.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                if ns not in new_sums:
                    new_coeffs = list(coeffs)
                    new_coeffs[i] = c
                    new_sums[ns] = new_coeffs
        sums_c = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Convert B sums to sorted list for binary search
    arr_b = sorted(sums_b.keys())
    b_min = arr_b[0]
    b_max = arr_b[-1]
    
    best_val = None
    best_a = None
    best_b = None
    best_coeffs_a = None
    best_coeffs_b = None
    
    # Sort A sums by absolute value to find small |a+b| faster
    a_list = sorted(sums_a.keys(), key=lambda a: abs(a))
    
    for a in a_list:
        if time.time() > deadline:
            break
        
        if best_val is not None and best_val == 1:
            break
        
        target = -a
        if target < b_min:
            val = a + b_min
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_min
                    best_coeffs_a = sums_a[a]
                    best_coeffs_b = sums_b[b_min]
        elif target > b_max:
            val = a + b_max
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b_max
                    best_coeffs_a = sums_a[a]
                    best_coeffs_b = sums_b[b_max]
        else:
            idx = bisect.bisect_left(arr_b, target)
            
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
                            best_coeffs_a = sums_a[a]
                            best_coeffs_b = sums_b[b]
                    if best_val == 1:
                        break
            if best_val == 1:
                break
    
    if best_a is not None and best_b is not None:
        # Now find best c in group C to minimize |best_a + best_b + c|
        ab_sum = best_a + best_b
        best_c_val = None
        best_c_abs = None
        best_coeffs_c = None
        
        for c_val, c_coeffs in sums_c.items():
            total = ab_sum + c_val
            if total == 0:
                continue
            abs_total = abs(total)
            if best_c_abs is None or abs_total < best_c_abs:
                best_c_abs = abs_total
                best_c_val = c_val
                best_coeffs_c = c_coeffs
        
        if best_c_val is not None:
            final_coeffs = best_coeffs_a + best_coeffs_b + best_coeffs_c
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
    
    return _fallback(W, n)

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
