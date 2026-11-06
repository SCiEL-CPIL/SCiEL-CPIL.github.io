import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    mid = n // 2  # 15
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_with_encoding(weights_part):
        """Generate all sums with base-3 encoding of coefficients"""
        # For each sum, store (sum_value, base3_encoding)
        # base3_encoding: digit i is the coefficient for weight i (0->-1, 1->0, 2->1)
        sums = [(0, 0)]  # (sum, encoding)
        for i, w in enumerate(weights_part):
            new_sums = []
            for s, enc in sums:
                # coefficient -1 -> digit 0
                new_sums.append((s - w, enc + 0 * (3 ** i)))
                # coefficient 0 -> digit 1
                new_sums.append((s, enc + 1 * (3 ** i)))
                # coefficient 1 -> digit 2
                new_sums.append((s + w, enc + 2 * (3 ** i)))
            sums = new_sums
        return sums
    
    # Generate sums for first half
    if time.time() > deadline - 20:
        return [1, -1] + [0] * 28
    
    sums1 = generate_sums_with_encoding(w1)
    
    if time.time() > deadline - 15:
        return [1, -1] + [0] * 28
    
    # Generate sums for second half
    sums2 = generate_sums_with_encoding(w2)
    
    if time.time() > deadline - 10:
        return [1, -1] + [0] * 28
    
    # Sort sums2 by value for binary search
    sums2_sorted = sorted(sums2, key=lambda x: x[0])
    sums2_values = np.array([s[0] for s in sums2_sorted], dtype=np.int64)
    
    # For each sum in sums1, find closest match in sums2
    best_abs = None
    best_s1_val = None
    best_s2_val = None
    best_enc1 = None
    best_enc2 = None
    
    sums1_values = np.array([s[0] for s in sums1], dtype=np.int64)
    sums1_encs = [s[1] for s in sums1]
    
    # Binary search for each target
    targets = -sums1_values
    idx = np.searchsorted(sums2_values, targets)
    
    # Check idx and idx-1
    for check_offset in [0, -1]:
        idx_c = np.clip(idx + check_offset, 0, len(sums2_values) - 1)
        s2_vals = sums2_values[idx_c]
        total_sums = sums1_values + s2_vals
        abs_sums = np.abs(total_sums)
        # Exclude zero sums
        abs_sums[total_sums == 0] = np.iinfo(np.int64).max
        
        if len(abs_sums) == 0:
            continue
            
        min_idx = np.argmin(abs_sums)
        min_val = abs_sums[min_idx]
        
        if best_abs is None or min_val < best_abs:
            best_abs = min_val
            best_s1_val = sums1_values[min_idx]
            best_s2_val = s2_vals[min_idx]
            best_enc1 = sums1_encs[min_idx]
            best_enc2 = sums2_sorted[idx_c[min_idx]][1]
    
    if best_abs is None or best_abs == np.iinfo(np.int64).max:
        return [1, -1] + [0] * 28
    
    # Decode coefficients from base-3 encodings
    def decode_coefficients(enc, n_digits):
        coeffs = []
        for i in range(n_digits):
            digit = enc % 3
            coeffs.append(digit - 1)  # 0->-1, 1->0, 2->1
            enc //= 3
        return coeffs
    
    c1 = decode_coefficients(best_enc1, mid)
    c2 = decode_coefficients(best_enc2, n - mid)
    
    coeffs = c1 + c2
    
    # Verify the result
    total = sum(c * w for c, w in zip(coeffs, weights))
    if total == 0:
        return [1, -1] + [0] * 28
    
    return coeffs
