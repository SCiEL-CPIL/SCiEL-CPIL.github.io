import math
import time
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    mid = 15
    
    def encode_coeff(c):
        if c == -1: return 0
        if c == 0: return 1
        if c == 1: return 2
        return 1 # default

    def decode_code(code, size):
        coeffs = [0] * size
        for i in range(size):
            val = (code >> (3 * i)) & 7
            if val == 0: coeffs[i] = -1
            elif val == 1: coeffs[i] = 0
            else: coeffs[i] = 1
        return coeffs

    # Generate sums for first half (indices 0..14)
    sums_a = {0: 0} # sum: code
    for i in range(mid):
        w = W[i]
        new_sums = {}
        for s, code in sums_a.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                nc = code | (encode_coeff(c) << (3 * i))
                if ns not in new_sums:
                    new_sums[ns] = nc
        sums_a = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
        
    # Generate sums for second half (indices 15..29)
    sums_b = {0: 0} # sum: code
    for i in range(mid, n):
        w = W[i]
        new_sums = {}
        for s, code in sums_b.items():
            for c in [-1, 0, 1]:
                ns = s + c * w
                # Shift index for encoding: local index is i - mid
                local_i = i - mid
                nc = code | (encode_coeff(c) << (3 * local_i))
                if ns not in new_sums:
                    new_sums[ns] = nc
        sums_b = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
        
    # Sort keys of B for binary search
    keys_b = sorted(sums_b.keys())
    b_min = keys_b[0]
    b_max = keys_b[-1]
    
    best_val = None
    best_a_sum = None
    best_b_sum = None
    
    # Iterate through A sums
    # To find small sums faster, we could sort A by abs value, but we need to check pairs.
    # Just iterating is fine.
    for a_sum in sums_a.keys():
        if time.time() > deadline:
            break
            
        if best_val is not None and best_val == 1:
            break
            
        target = -a_sum
        if target < b_min:
            val = a_sum + b_min
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a_sum = a_sum
                    best_b_sum = b_min
        elif target > b_max:
            val = a_sum + b_max
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a_sum = a_sum
                    best_b_sum = b_max
        else:
            idx = bisect.bisect_left(keys_b, target)
            
            for candidate_idx in [idx, idx - 1]:
                if 0 <= candidate_idx < len(keys_b):
                    b_sum = keys_b[candidate_idx]
                    val = a_sum + b_sum
                    if val != 0:
                        abs_val = abs(val)
                        if best_val is None or abs_val < best_val:
                            best_val = abs_val
                            best_a_sum = a_sum
                            best_b_sum = b_sum
                    if best_val == 1:
                        break
            if best_val == 1:
                break
                
    if best_a_sum is not None and best_b_sum is not None:
        code_a = sums_a[best_a_sum]
        code_b = sums_b[best_b_sum]
        
        coeffs_a = decode_code(code_a, mid)
        coeffs_b = decode_code(code_b, mid)
        
        final_coeffs = coeffs_a + coeffs_b
        
        # Verify
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
