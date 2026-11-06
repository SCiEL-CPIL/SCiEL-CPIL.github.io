import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Calculate LCM of 1..30
    D = 1
    for i in range(1, 31):
        D = D * i // gcd(D, i)
        
    # 2. Generate sums and codes for first half (1..15)
    # Weights for first half
    weights_A = [D // i for i in range(1, 16)]
    
    sums_A = np.array([0], dtype=np.int64)
    codes_A = np.array([0], dtype=np.int32)
    
    for i, w in enumerate(weights_A):
        # Expand sums: s-w, s, s+w
        # Expand codes: c*3+0, c*3+1, c*3+2
        # Order matters to keep alignment
        sums_A = np.concatenate([sums_A - w, sums_A, sums_A + w])
        codes_A = np.concatenate([codes_A * 3, codes_A * 3 + 1, codes_A * 3 + 2])
        
    # 3. Generate sums and codes for second half (16..30)
    weights_B = [D // i for i in range(16, 31)]
    
    sums_B = np.array([0], dtype=np.int64)
    codes_B = np.array([0], dtype=np.int32)
    
    for i, w in enumerate(weights_B):
        sums_B = np.concatenate([sums_B - w, sums_B, sums_B + w])
        codes_B = np.concatenate([codes_B * 3, codes_B * 3 + 1, codes_B * 3 + 2])
        
    if time.time() > deadline - 5:
        return [0] * 28 + [1, -1]

    # 4. Sort A and B by sums
    order_A = np.argsort(sums_A)
    sums_A_sorted = sums_A[order_A]
    codes_A_sorted = codes_A[order_A]
    
    order_B = np.argsort(sums_B)
    sums_B_sorted = sums_B[order_B]
    codes_B_sorted = codes_B[order_B]
    
    # 5. Find min |a + b| != 0
    # For each b in B, find closest a in A to -b
    # Use searchsorted on sums_A_sorted
    
    targets = -sums_B_sorted
    indices = np.searchsorted(sums_A_sorted, targets)
    
    # Candidates are at indices and indices-1
    # Handle boundaries
    idx1 = np.clip(indices, 0, len(sums_A_sorted) - 1)
    idx2 = np.clip(indices - 1, 0, len(sums_A_sorted) - 1)
    
    # Calculate differences
    diff1 = sums_A_sorted[idx1] + sums_B_sorted
    diff2 = sums_A_sorted[idx2] + sums_B_sorted
    
    abs_diff1 = np.abs(diff1)
    abs_diff2 = np.abs(diff2)
    
    # We want min abs_diff where diff != 0
    # Set zeros to infinity
    abs_diff1[diff1 == 0] = np.inf
    abs_diff2[diff2 == 0] = np.inf
    
    min_abs_diff = np.minimum(abs_diff1, abs_diff2)
    min_idx = np.argmin(min_abs_diff)
    
    # Check if the minimum is still inf (should not happen unless all sums are 0, which is impossible)
    if min_abs_diff[min_idx] == np.inf:
        return [0] * 28 + [1, -1]

    # Determine which candidate (1 or 2) was the min
    if abs_diff1[min_idx] <= abs_diff2[min_idx]:
        best_a_idx = idx1[min_idx]
        best_b_idx = min_idx
    else:
        best_a_idx = idx2[min_idx]
        best_b_idx = min_idx
        
    # Retrieve codes
    code_A = codes_A_sorted[best_a_idx]
    code_B = codes_B_sorted[best_b_idx]
    
    # 6. Decode codes to coefficients
    coeffs = [0] * 30
    
    # Decode A (1..15)
    # c1 is MSB, c15 is LSB
    for i in range(14, -1, -1):
        digit = code_A % 3
        code_A //= 3
        coeffs[i] = digit - 1
        
    # Decode B (16..30)
    # c16 is MSB, c30 is LSB
    # Index 15 in coeffs corresponds to 1/16
    for i in range(29, 14, -1):
        digit = code_B % 3
        code_B //= 3
        coeffs[i] = digit - 1
        
    return coeffs
