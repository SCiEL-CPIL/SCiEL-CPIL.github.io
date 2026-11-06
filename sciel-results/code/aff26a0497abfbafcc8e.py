import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n = 30
    # Compute LCM(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n+1))
    
    half = n // 2  # 15
    first_half = list(range(1, half+1))       # 1..15
    second_half = list(range(half+1, n+1))    # 16..30
    
    def gen_sums_codes(indices):
        """Generate all 3^len(indices) possible sums and their base-3 codes."""
        vals = [L // i for i in indices]
        n_items = len(indices)
        
        # Start with sum=0, code=0
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        power = 1
        for v in vals:
            N = len(sums)
            # New sums: sum-v, sum, sum+v
            # Corresponding codes: code - power, code, code + power
            sums_new = np.empty(3 * N, dtype=np.int64)
            codes_new = np.empty(3 * N, dtype=np.int64)
            
            sums_new[0:N] = sums - v
            sums_new[N:2*N] = sums
            sums_new[2*N:3*N] = sums + v
            
            codes_new[0:N] = codes - power
            codes_new[N:2*N] = codes
            codes_new[2*N:3*N] = codes + power
            
            sums = sums_new
            codes = codes_new
            power *= 3
            
            # Check deadline
            if time.time() > deadline:
                return None, None
        
        return sums, codes

    # Generate for first half
    sums_A, codes_A = gen_sums_codes(first_half)
    if sums_A is None:
        # Fallback: simple heuristic
        # Try to find a small sum by greedy or random
        return [1, -1] + [0]*28
        
    if time.time() > deadline:
        return [1, -1] + [0]*28

    # Generate for second half
    sums_B, codes_B = gen_sums_codes(second_half)
    if sums_B is None:
        return [1, -1] + [0]*28
        
    if time.time() > deadline:
        return [1, -1] + [0]*28

    # Sort sums_B and keep codes_B aligned
    idx_sort = np.argsort(sums_B)
    sums_B_sorted = sums_B[idx_sort]
    codes_B_sorted = codes_B[idx_sort]
    
    N_A = len(sums_A)
    N_B = len(sums_B_sorted)
    
    # Find best pair
    # For each s_a in sums_A, find s_b in sums_B_sorted closest to -s_a
    target = -sums_A
    positions = np.searchsorted(sums_B_sorted, target)
    
    # Candidate 1: position
    pos1 = np.clip(positions, 0, N_B-1)
    s_b_1 = sums_B_sorted[pos1]
    c_b_1 = codes_B_sorted[pos1]
    total_1 = sums_A + s_b_1
    
    # Candidate 2: position - 1
    pos2 = np.clip(positions - 1, 0, N_B-1)
    s_b_2 = sums_B_sorted[pos2]
    c_b_2 = codes_B_sorted[pos2]
    total_2 = sums_A + s_b_2
    
    # Combine
    all_totals = np.concatenate([total_1, total_2])
    all_codes_A = np.concatenate([codes_A, codes_A])
    all_codes_B = np.concatenate([c_b_1, c_b_2])
    
    # Filter non-zero
    mask = all_totals != 0
    if not np.any(mask):
        # If all are zero (unlikely), return a valid non-zero
        return [1, -1] + [0]*28
        
    valid_totals = all_totals[mask]
    valid_codes_A = all_codes_A[mask]
    valid_codes_B = all_codes_B[mask]
    
    # Find min absolute value
    abs_totals = np.abs(valid_totals)
    min_idx = np.argmin(abs_totals)
    
    best_code_A = valid_codes_A[min_idx]
    best_code_B = valid_codes_B[min_idx]
    
    # Decode
    def decode_code(code, n_digits):
        coeffs = [0] * n_digits
        code = int(code)
        for i in range(n_digits):
            d = code % 3
            if d > 1:
                d -= 3
            coeffs[i] = d
            code = (code - d) // 3
        return coeffs
        
    coeffs1 = decode_code(best_code_A, half)
    coeffs2 = decode_code(best_code_B, n - half)
    
    result = coeffs1 + coeffs2
    
    # Verify
    if len(result) != 30:
        return [1, -1] + [0]*28
    if not all(c in (-1, 0, 1) for c in result):
        return [1, -1] + [0]*28
        
    total = sum(result[i] / (i+1) for i in range(30))
    if total == 0:
        return [1, -1] + [0]*28
        
    return result
