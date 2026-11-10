import time
from math import gcd
from fractions import Fraction
import bisect
import random

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    n = 30
    
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    # weights[i] = L // (i+1) for i in 0..29
    weights = [L // (i + 1) for i in range(n)]
    
    # Meet-in-the-middle: split into two halves of 15
    half = 15
    left_w = weights[:half]
    right_w = weights[half:]
    
    # Generate all 3^15 = 14348907 sums for each half
    # For each sum, store one representative coefficient vector (as a tuple of 15 values in {-1,0,1})
    # We want to minimize |left_sum + right_sum| where the total != 0
    
    # To save memory, we only need to store the sums and one representative mask
    # Actually, we need to reconstruct coefficients, so let's store the sum -> list of (coeff_tuple)
    # But 3^15 is ~14.3M, which is a lot of memory if we store tuples.
    
    # Better approach: generate all sums for the right half, sort them, and for each left sum,
    # binary search for the closest right sum.
    
    # Generate right sums: each is a sum of c_i * right_w[i] for c_i in {-1,0,1}
    # We need to track which coefficients produce each sum to reconstruct.
    
    # Let's generate all right sums with their coefficient patterns.
    # 3^15 = 14348907 entries. Each entry: (sum_value, coeff_pattern)
    # coeff_pattern can be stored as a base-3 integer (15 digits, each 0,1,2 representing -1,0,1)
    
    right_sums = []  # list of (sum_val, base3_code)
    
    # Iterative generation
    # Start with [(0, 0)]
    states = [(0, 0)]
    for i in range(half):
        w = right_w[i]
        new_states = []
        for s, code in states:
            new_states.append((s, code))           # coeff = 0
            new_states.append((s + w, code + 1 * (3 ** i)))   # coeff = 1
            new_states.append((s - w, code + 2 * (3 ** i)))   # coeff = -1
        states = new_states
    
    # Now states has 3^15 entries
    # Convert to list of (sum_val, code)
    # Sort by sum_val
    right_sums = states
    right_sums.sort(key=lambda x: x[0])
    
    # Extract sorted values for binary search
    right_vals = [x[0] for x in right_sums]
    
    # Now generate left sums and for each, find best matching right sum
    best_abs = None
    best_left_code = None
    best_right_code = None
    best_left_sum = None
    best_right_sum = None
    
    # Generate left sums iteratively
    left_states = [(0, 0)]
    for i in range(half):
        w = left_w[i]
        new_states = []
        for s, code in left_states:
            new_states.append((s, code))           # coeff = 0
            new_states.append((s + w, code + 1 * (3 ** i)))   # coeff = 1
            new_states.append((s - w, code + 2 * (3 ** i)))   # coeff = -1
        left_states = new_states
        # Time guard
        if time.time() > DEADLINE - 30.0:
            break
    
    # For each left state, find best right match
    for s_l, code_l in left_states:
        target = -s_l
        # Binary search in right_vals
        idx = bisect.bisect_left(right_vals, target)
        candidates = []
        for pos in (idx - 1, idx, idx + 1):
            if 0 <= pos < len(right_vals):
                candidates.append(pos)
        
        for pos in candidates:
            s_r = right_vals[pos]
            total = s_l + s_r
            if total == 0:
                continue
            a = abs(total)
            if best_abs is None or a < best_abs:
                best_abs = a
                best_left_code = code_l
                best_right_code = right_sums[pos][1]
                best_left_sum = s_l
                best_right_sum = s_r
        
        # Time guard
        if time.time() > DEADLINE - 5.0:
            break
    
    # Reconstruct coefficients from base-3 codes
    coeffs = [0] * n
    
    # Left half: code_l encodes coefficients for positions 0..14
    if best_left_code is not None:
        for i in range(half):
            digit = (best_left_code // (3 ** i)) % 3
            if digit == 1:
                coeffs[i] = 1
            elif digit == 2:
                coeffs[i] = -1
            else:
                coeffs[i] = 0
    
    # Right half: code_r encodes coefficients for positions 15..29
    if best_right_code is not None:
        for i in range(half):
            digit = (best_right_code // (3 ** i)) % 3
            if digit == 1:
                coeffs[half + i] = 1
            elif digit == 2:
                coeffs[half + i] = -1
            else:
                coeffs[half + i] = 0
    
    # Verify the sum is nonzero
    total_frac = Fraction(0, 1)
    for i in range(n):
        if coeffs[i] != 0:
            total_frac += Fraction(coeffs[i], i + 1)
    
    if total_frac == 0:
        # Fallback: find a simple nonzero sum
        # Try flipping one coefficient
        for i in range(n):
            old = coeffs[i]
            for new_val in (-1, 0, 1):
                if new_val == old:
                    continue
                coeffs[i] = new_val
                t = Fraction(0, 1)
                for j in range(n):
                    if coeffs[j] != 0:
                        t += Fraction(coeffs[j], j + 1)
                if t != 0:
                    break
            if total_frac != 0:
                break
            coeffs[i] = old
        # If still zero, use a simple known nonzero
        coeffs = [0] * 28 + [1, -1]
        return coeffs
    
    return coeffs
