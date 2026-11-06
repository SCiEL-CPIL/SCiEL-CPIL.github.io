import numpy as np
from math import gcd
from functools import reduce

def construct():
    n = 30
    # LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n+1))
    
    # Values: L/i for i=1..n (use Python ints to avoid overflow)
    vals = [L // i for i in range(1, n+1)]
    
    # Split into two halves
    mid = n // 2
    left_vals = vals[:mid]
    right_vals = vals[mid:]
    
    # Function to generate all 3^k signed sums using Python ints
    def generate_sums(values):
        sums = [0]
        for v in values:
            sums = sums + [s + v for s in sums] + [s - v for s in sums]
        return sums

    # Generate sums for left and right halves
    left_sums = generate_sums(left_vals)
    right_sums = generate_sums(right_vals)
    
    # Sort right sums and keep track of original indices
    right_indices = sorted(range(len(right_sums)), key=lambda i: right_sums[i])
    sorted_right_sums = [right_sums[i] for i in right_indices]
    
    best_abs = float('inf')
    best_left_idx = -1
    best_right_idx = -1
    
    # Use binary search to find closest match for each left sum
    import bisect
    
    for li, ls in enumerate(left_sums):
        target = -ls
        # Find insertion point in sorted_right_sums
        pos = bisect.bisect_left(sorted_right_sums, target)
        
        # Check candidates at pos and pos-1
        for cand_pos in [pos - 1, pos]:
            if 0 <= cand_pos < len(sorted_right_sums):
                total = ls + sorted_right_sums[cand_pos]
                if total != 0:
                    abs_total = abs(total)
                    if abs_total < best_abs:
                        best_abs = abs_total
                        best_left_idx = li
                        best_right_idx = right_indices[cand_pos]
    
    if best_left_idx == -1:
        # Fallback
        return [0] * 28 + [1, -1]
    
    # Decode base-3 indices to coefficients
    def decode_index(idx, num_vars):
        coeffs = [0] * num_vars
        for i in range(num_vars):
            d = idx % 3
            idx //= 3
            if d == 0:
                coeffs[i] = 0
            elif d == 1:
                coeffs[i] = 1
            else:  # d == 2
                coeffs[i] = -1
        return coeffs
    
    left_coeffs = decode_index(best_left_idx, mid)
    right_coeffs = decode_index(best_right_idx, n - mid)
    
    full_coeffs = left_coeffs + right_coeffs
    
    # Verify
    total = 0
    for i, c in enumerate(full_coeffs):
        if c != 0:
            total += c * (L // (i + 1))
    
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return full_coeffs
