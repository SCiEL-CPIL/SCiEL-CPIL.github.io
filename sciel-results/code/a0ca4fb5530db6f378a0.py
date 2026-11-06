import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = np.array([H // i for i in range(1, n + 1)], dtype=np.int64)
    
    mid = 15
    
    def generate_sums_and_codes(weights):
        # weights is a numpy array of size m
        m = len(weights)
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i in range(m):
            w = weights[i]
            # Create new sums: s - w, s, s + w
            # We need to interleave or just concatenate and sort later?
            # Concatenation is fine.
            sums_new = np.concatenate([sums - w, sums, sums + w])
            
            # Create new codes
            # Code bits for index i:
            # -1 -> 0, 0 -> 1, 1 -> 2
            # We need to set the 3 bits at position 3*i.
            # The existing codes have bits for 0..i-1.
            # We need to shift the new digit into place.
            # Wait, the code structure is:
            # digit 0 at bits 0-2
            # digit 1 at bits 3-5
            # ...
            # So for index i, we write to bits 3*i to 3*i+2.
            
            # We have 3 copies of the old codes.
            # Copy 1 (for -1): digit 0. Code = old_code | (0 << 3*i) = old_code
            # Copy 2 (for 0): digit 1. Code = old_code | (1 << 3*i)
            # Copy 3 (for 1): digit 2. Code = old_code | (2 << 3*i)
            
            c1 = codes
            c2 = codes | (1 << (3 * i))
            c3 = codes | (2 << (3 * i))
            
            codes_new = np.concatenate([c1, c2, c3])
            
            sums = sums_new
            codes = codes_new
            
        return sums, codes

    # Generate for first half
    sums_a, codes_a = generate_sums_and_codes(W[:mid])
    if time.time() > deadline:
        return _fallback(W, n)

    # Generate for second half
    sums_b, codes_b = generate_sums_and_codes(W[mid:])
    if time.time() > deadline:
        return _fallback(W, n)

    # Sort A and B
    idx_a = np.argsort(sums_a)
    sums_a_sorted = sums_a[idx_a]
    codes_a_sorted = codes_a[idx_a]
    
    idx_b = np.argsort(sums_b)
    sums_b_sorted = sums_b[idx_b]
    codes_b_sorted = codes_b[idx_b]
    
    # Two-pointer search for T = 1, 2, ...
    # We want min |a+b| != 0.
    # So we check T = 1, -1, 2, -2, ...
    # Or just check T = 1, 2, ... and assume symmetry?
    # The set of sums is symmetric?
    # If c is a solution, -c is also a solution (coeffs -1,0,1 are symmetric).
    # So if T is achievable, -T is achievable.
    # So we just need to find the smallest positive T.
    
    len_a = len(sums_a_sorted)
    len_b = len(sums_b_sorted)
    
    best_val = None
    best_code_a = None
    best_code_b = None
    
    # We will check T = 1, 2, ... up to a limit
    # If not found, we fallback to finding closest to 0 (which might be 0, then we need next)
    # But given the density, T=1 should be found.
    
    max_T = 1000
    for T in range(1, max_T + 1):
        if time.time() > deadline:
            break
            
        i = 0
        j = len_b - 1
        
        found = False
        while i < len_a and j >= 0:
            s = sums_a_sorted[i] + sums_b_sorted[j]
            if s == T:
                best_val = T
                best_code_a = codes_a_sorted[i]
                best_code_b = codes_b_sorted[j]
                found = True
                break
            elif s < T:
                i += 1
            else:
                j -= 1
                
        if found:
            break
            
    if best_val is not None:
        # Decode
        coeffs_a = decode_code(best_code_a, mid)
        coeffs_b = decode_code(best_code_b, mid)
        final_coeffs = coeffs_a + coeffs_b
        
        # Verify
        total = sum(c * W[i] for i, c in enumerate(final_coeffs))
        if total != 0:
            return final_coeffs
            
    # Fallback if T not found in range or error
    return _fallback(W, n)

def decode_code(code, size):
    coeffs = [0] * size
    for i in range(size):
        val = (code >> (3 * i)) & 7
        if val == 0: coeffs[i] = -1
        elif val == 1: coeffs[i] = 0
        else: coeffs[i] = 1
    return coeffs

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
