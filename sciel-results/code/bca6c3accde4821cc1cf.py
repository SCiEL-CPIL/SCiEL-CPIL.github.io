import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // (i + 1) for i in range(30)]
    
    # Split indices
    left_indices = list(range(15))
    right_indices = list(range(15, 30))
    
    left_weights = [weights[i] for i in left_indices]
    right_weights = [weights[i] for i in right_indices]
    
    # Generate all sums for left half
    # State: (sum, code)
    # code bits: 2*i, 2*i+1 for index i in left half
    # 0: 0, 1: 1, 2: -1 (mapping to coeff 0, 1, -1)
    
    def generate_sums(weights_list, num_items):
        sums = [(0, 0)]
        for i in range(num_items):
            w = weights_list[i]
            new_sums = []
            bit_offset = 2 * i
            for s, c in sums:
                # 0
                new_sums.append((s, c))
                # 1
                new_sums.append((s + w, c | (1 << bit_offset)))
                # -1
                new_sums.append((s - w, c | (2 << bit_offset)))
            sums = new_sums
        return sums

    # Check time
    if time.time() > deadline - 30:
        # Fallback to heuristic if generation is too slow? 
        # But 14M items should be okay.
        pass

    left_sums = generate_sums(left_weights, 15)
    # Sort by sum
    left_sums.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    left_sum_values = [x[0] for x in left_sums]
    
    best_sum_val = float('inf')
    best_code_left = 0
    best_code_right = 0
    
    # Generate right sums and check
    # We can generate right sums on the fly to save memory
    right_sums_gen = generate_sums(right_weights, 15)
    
    import bisect
    
    for s_r, c_r in right_sums_gen:
        target = -s_r
        # Find index in left_sum_values closest to target
        idx = bisect.bisect_left(left_sum_values, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(left_sum_values):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for i in candidates:
            s_l = left_sum_values[i]
            c_l = left_sums[i][1]
            total_sum = s_l + s_r
            if total_sum == 0:
                continue
            if abs(total_sum) < best_sum_val:
                best_sum_val = abs(total_sum)
                best_code_left = c_l
                best_code_right = c_r

    # Reconstruct coefficients
    coeffs = [0] * 30
    for i in range(15):
        val = (best_code_left >> (2 * i)) & 3
        if val == 0: coeffs[i] = 0
        elif val == 1: coeffs[i] = 1
        else: coeffs[i] = -1
        
    for i in range(15):
        val = (best_code_right >> (2 * i)) & 3
        if val == 0: coeffs[15+i] = 0
        elif val == 1: coeffs[15+i] = 1
        else: coeffs[15+i] = -1
        
    # Verify sum is nonzero
    total = sum(c * w for c, w in zip(coeffs, weights))
    if total == 0:
        # Should not happen due to check, but just in case
        coeffs[0] = 1
        total = weights[0]
        
    return coeffs
