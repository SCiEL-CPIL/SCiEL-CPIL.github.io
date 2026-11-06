import time
import numpy as np
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))

    # W[i] = L // (i+1), so sum(c[i]/(i+1)) = (1/L) * sum(c[i]*W[i])
    # We want to minimize |sum(c[i]*W[i])| subject to it being nonzero
    W = [L // (i + 1) for i in range(n)]

    # Split into two halves for meet-in-the-middle
    mid = n // 2
    w1 = W[:mid]
    w2 = W[mid:]

    def generate_sums_with_coeffs(weights_half):
        """
        Generate all possible sums and their coefficient patterns for a half.
        Returns a dict mapping sum -> list of coefficients (first found).
        """
        n_half = len(weights_half)
        sums_dict = {0: [0] * n_half}
        
        for i, w in enumerate(weights_half):
            new_sums_dict = {}
            for s, coeffs in sums_dict.items():
                # Try -1
                ns = s - w
                new_coeffs = coeffs.copy()
                new_coeffs[i] = -1
                if ns not in new_sums_dict:
                    new_sums_dict[ns] = new_coeffs
                # Try 0
                ns = s
                new_coeffs = coeffs.copy()
                new_coeffs[i] = 0
                if ns not in new_sums_dict:
                    new_sums_dict[ns] = new_coeffs
                # Try 1
                ns = s + w
                new_coeffs = coeffs.copy()
                new_coeffs[i] = 1
                if ns not in new_sums_dict:
                    new_sums_dict[ns] = new_coeffs
            sums_dict = new_sums_dict
            
        return sums_dict

    # Generate sums for both halves
    sums1_dict = generate_sums_with_coeffs(w1)
    sums2_dict = generate_sums_with_coeffs(w2)

    # Extract keys for binary search
    sums1_list = list(sums1_dict.keys())
    sums2_list = sorted(sums2_dict.keys())
    
    len_s2 = len(sums2_list)
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # For each s1, we want s2 close to -s1.
    
    best_abs_val = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    # Iterate over sums1
    for s1 in sums1_list:
        if time.time() > deadline:
            break
            
        target = -s1
        # Find insertion point in sums2_list
        idx = np.searchsorted(sums2_list, target)
        
        # Check neighbors: idx-1, idx, idx+1
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len_s2:
            candidates.append(idx)
        if idx + 1 < len_s2:
            candidates.append(idx + 1)
            
        for c_idx in candidates:
            s2 = sums2_list[c_idx]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs_val:
                    best_abs_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    # If we found 1, that's the minimum possible non-zero integer
                    if best_abs_val == 1:
                        break
        if best_abs_val == 1:
            break
            
    if best_abs_val == float('inf'):
        # Fallback: should not happen, but just in case
        # Try to find any non-zero sum
        for s1 in sums1_list:
            for s2 in sums2_list[:10]:
                if s1 + s2 != 0:
                    best_abs_val = abs(s1 + s2)
                    best_s1 = s1
                    best_s2 = s2
                    break
            if best_abs_val != float('inf'):
                break
                
    if best_abs_val == float('inf'):
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    coeffs1 = sums1_dict[best_s1]
    coeffs2 = sums2_dict[best_s2]
    
    coeffs = coeffs1 + coeffs2
    
    # Verify
    # val = sum(coeffs[i]/(i+1) for i in range(n))
    # assert val != 0
    # assert abs(val) == Fraction(best_abs_val, L)
    
    return coeffs
