import numpy as np
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i
    weights = np.array([L // i for i in range(1, N + 1)], dtype=np.int64)
    
    # Split into two halves of 15
    W1 = weights[:15]
    W2 = weights[15:]
    
    def gen_sums_for_group(weights_group):
        """
        Generates all sums and codes for a group of weights.
        Returns numpy arrays of sums and codes.
        """
        n = len(weights_group)
        # Split group into two subgroups for faster generation
        # e.g., 7 and 8
        n1 = n // 2
        n2 = n - n1
        
        W1_sub = weights_group[:n1]
        W2_sub = weights_group[n1:]
        
        # Generate sums for subgroups
        sums1, codes1 = _gen_sums_simple(W1_sub)
        sums2, codes2 = _gen_sums_simple(W2_sub)
        
        # Combine
        # Sums are s1 + s2
        # Codes are c1 * 3^n2 + c2
        
        # Vectorized combination
        # sums1 shape (3^n1,), sums2 shape (3^n2,)
        # Result shape (3^n1 * 3^n2,)
        
        # To avoid huge memory spikes or slow loops, we can do:
        # s_combined = sums1[:, None] + sums2[None, :]
        # But this creates a 2D array of size 3^n1 x 3^n2.
        # 3^7 = 2187, 3^8 = 6561.
        # 2187 * 6561 = 14M.
        # 14M int64s is ~112 MB. Acceptable.
        
        s_combined = (sums1[:, None] + sums2[None, :]).flatten()
        
        # Codes:
        # c1 is repeated n2 times, c2 is repeated n1 times
        # c_combined = c1[:, None] * (3**n2) + c2[None, :]
        
        pow3_n2 = 3 ** n2
        c_combined = (codes1[:, None] * pow3_n2 + codes2[None, :]).flatten()
        
        return s_combined, c_combined

    def _gen_sums_simple(weights_sub):
        """
        Generates sums and codes for a small group (size <= 8).
        Uses iterative base-3 or product.
        """
        n = len(weights_sub)
        if n == 0:
            return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)
        
        # Use itertools.product for simplicity and speed in C
        import itertools
        # Map -1, 0, 1 to 0, 1, 2 for code
        # coeffs = [-1, 0, 1]
        # code digit = coeff + 1
        
        sums = []
        codes = []
        
        # This loop is 3^n iterations. Max 3^8 = 6561. Very fast.
        for prod in itertools.product([-1, 0, 1], repeat=n):
            s = 0
            c = 0
            for i, val in enumerate(prod):
                s += val * weights_sub[i]
                # code digit: 0->-1, 1->0, 2->1
                # val is -1, 0, 1. digit = val + 1.
                # code = sum(digit_i * 3^i)
                # Note: index i corresponds to weight i.
                # In our code representation, digit i is at position i.
                # So code = sum((val+1) * 3^i)
                c += (val + 1) * (3 ** i)
            sums.append(s)
            codes.append(c)
            
        return np.array(sums, dtype=np.int64), np.array(codes, dtype=np.int64)

    # Generate sums for both halves
    sums1, codes1 = gen_sums_for_group(W1)
    sums2, codes2 = gen_sums_for_group(W2)
    
    # Sort sums2 and keep track of codes2
    # We need to sort sums2, and permute codes2 accordingly.
    sorted_indices = np.argsort(sums2)
    sums2_sorted = sums2[sorted_indices]
    codes2_sorted = codes2[sorted_indices]
    
    best_abs_sum = float('inf')
    best_code1 = -1
    best_code2 = -1
    
    # Search
    # For each sum in sums1, find best match in sums2_sorted
    n2 = len(sums2_sorted)
    
    for i in range(len(sums1)):
        s1 = sums1[i]
        target = -s1
        
        # Binary search in sums2_sorted
        k = np.searchsorted(sums2_sorted, target)
        
        # Check neighbors around k
        # k is the first index where sums2_sorted[k] >= target
        # Candidates: k-1, k, k+1, k-2, k+2
        
        candidates = []
        for offset in [-2, -1, 0, 1, 2]:
            idx = k + offset
            if 0 <= idx < n2:
                candidates.append(idx)
        
        for idx in candidates:
            s2 = sums2_sorted[idx]
            total_sum = s1 + s2
            
            if total_sum != 0:
                abs_sum = abs(total_sum)
                if abs_sum < best_abs_sum:
                    best_abs_sum = abs_sum
                    best_code1 = codes1[i]
                    best_code2 = codes2_sorted[idx]
                    
                    # Early termination if we find 1?
                    # Minimum non-zero integer is 1.
                    if best_abs_sum == 1:
                        # We can't return immediately because we are in a loop
                        # but we can break out
                        break
        
        if best_abs_sum == 1:
            break
            
    if best_abs_sum == float('inf'):
        # Fallback: should not happen
        return [1] + [0] * (N - 1)
        
    # Reconstruct coefficients
    # code1 is for W1 (15 weights)
    # code2 is for W2 (15 weights)
    
    coeffs = [0] * N
    
    # Decode code1
    for i in range(15):
        digit = (best_code1 // (3 ** i)) % 3
        coeffs[i] = digit - 1
        
    # Decode code2
    for i in range(15):
        digit = (best_code2 // (3 ** i)) % 3
        coeffs[15 + i] = digit - 1
        
    return coeffs
