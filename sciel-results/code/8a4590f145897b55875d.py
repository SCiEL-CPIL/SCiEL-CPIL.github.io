"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from fractions import Fraction
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # Safety margin
    N = 30
    L = 2329089562800  # lcm(1..30)
    
    # Weights w_i = L / i
    weights = [L // i for i in range(1, N + 1)]
    
    # Split into two halves
    N1 = 15
    N2 = 15
    
    # Generate all possibilities for the first half
    # coeffs are -1, 0, 1
    # We represent the state as a base-3 integer code
    # code = sum (c_i + 1) * (3 ** i)
    
    # Precompute powers of 3
    pow3 = [1] * (N1 + 1)
    for i in range(1, N1 + 1):
        pow3[i] = pow3[i-1] * 3
    
    # List to store (sum_value, code) for first half
    list_A = []
    
    # Iterate through all 3^15 combinations
    # We can use a recursive generator or iterative counting
    # Iterative counting is faster in Python if done carefully, but recursion is clearer.
    # Given 3^15 is ~14M, recursion depth 15 is fine.
    
    def gen_combos(start_idx, end_idx, current_sum, current_code, weights_subset):
        if start_idx == end_idx:
            list_A.append((current_sum, current_code))
            return
        
        # Option 0: coeff = -1
        gen_combos(start_idx + 1, end_idx, current_sum - weights_subset[start_idx], current_code + 0 * pow3[start_idx], weights_subset)
        # Option 1: coeff = 0
        gen_combos(start_idx + 1, end_idx, current_sum, current_code + 1 * pow3[start_idx], weights_subset)
        # Option 2: coeff = 1
        gen_combos(start_idx + 1, end_idx, current_sum + weights_subset[start_idx], current_code + 2 * pow3[start_idx], weights_subset)

    # To avoid recursion overhead for 14M calls, let's use an iterative approach or a more efficient generation.
    # Actually, 14M recursive calls might be slow in Python (stack depth 15, but 14M leaves).
    # Let's try to generate them iteratively.
    
    list_A.clear()
    # Initialize list_A with all combinations
    # We can use product from itertools, but it creates tuples.
    # Let's stick to a loop over range(3**15) and decode.
    # 3**15 = 14348907
    # This is fast.
    
    max_code_A = 3 ** N1
    for code in range(max_code_A):
        s = 0
        temp_code = code
        for i in range(N1):
            digit = temp_code % 3
            temp_code //= 3
            c = digit - 1  # -1, 0, 1
            if c != 0:
                s += c * weights[i]
        list_A.append((s, code))
        
    # Sort list_A by sum value
    list_A.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums_A = [item[0] for item in list_A]
    
    # Now iterate through all possibilities for the second half
    best_abs_val = float('inf')
    best_code_A = None
    best_code_B = None
    
    max_code_B = 3 ** N2
    weights_B = weights[N1:N]
    
    for code_B in range(max_code_B):
        s_B = 0
        temp_code = code_B
        for i in range(N2):
            digit = temp_code % 3
            temp_code //= 3
            c = digit - 1
            if c != 0:
                s_B += c * weights_B[i]
                
        # We want to find s_A such that |s_A + s_B| is minimized and non-zero.
        # Target s_A = -s_B
        # Use bisect to find insertion point in sums_A
        import bisect
        pos = bisect.bisect_left(sums_A, -s_B)
        
        # Check neighbors around pos
        # Candidates: pos-1, pos, pos+1 (within bounds)
        # We need to check a small window to handle duplicates and find the closest non-zero sum
        
        for p in range(max(0, pos-5), min(len(list_A), pos+6)):
            s_A, c_A = list_A[p]
            total = s_A + s_B
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs_val:
                best_abs_val = abs_total
                best_code_A = c_A
                best_code_B = code_B
                
                # If we find 1, it's the minimum possible non-zero integer sum (since L is integer)
                # Wait, the sum is an integer multiple of 1/L? No, the sum is s_A/L + s_B/L = (s_A+s_B)/L.
                # The numerator is s_A+s_B. We are minimizing |s_A+s_B|.
                # The smallest non-zero integer is 1.
                if best_abs_val == 1:
                    break
        
        if best_abs_val == 1:
            break
            
    # Reconstruct coefficients
    coeffs = [0] * N
    
    # Decode code_A
    temp_code = best_code_A
    for i in range(N1):
        digit = temp_code % 3
        temp_code //= 3
        coeffs[i] = digit - 1
        
    # Decode code_B
    temp_code = best_code_B
    for i in range(N2):
        digit = temp_code % 3
        temp_code //= 3
        coeffs[N1 + i] = digit - 1
        
    # Verify
    # The task requires returning exactly 30 integer coefficients from {-1, 0, 1}.
    # The signed sum must be nonzero.
    
    return coeffs
# EVOLVE-BLOCK-END
