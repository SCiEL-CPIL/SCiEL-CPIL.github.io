import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin vs 90s kill

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i

    N = 30
    # Split into two halves of 15
    N1 = 15
    N2 = 15
    
    # A1 corresponds to i=1..15
    # A2 corresponds to i=16..30
    A1 = np.array([A[i] for i in range(1, N1 + 1)], dtype=np.int64)
    A2 = np.array([A[i] for i in range(N1 + 1, N + 1)], dtype=np.int64)

    # Generate all 3^15 combinations for first half
    # Using unravel_index to get base-3 digits
    # Shape: (3^15, 15)
    
    # To save memory, we can generate in chunks or just use int8
    # 3^15 = 14348907
    # 14M * 15 * 1 byte = 215 MB. OK.
    
    def generate_coeffs(n_vars):
        # Returns array of shape (3^n_vars, n_vars) with values -1, 0, 1
        size = 3 ** n_vars
        # unravel_index returns tuple of arrays
        # We stack them to get shape (size, n_vars)
        # Note: unravel_index order is C-order (last index changes fastest)
        # This is fine.
        indices = np.unravel_index(np.arange(size), (3,) * n_vars)
        coeffs = np.stack(indices, axis=-1)
        coeffs = coeffs.astype(np.int8)
        coeffs = coeffs - 1  # Map 0,1,2 to -1,0,1
        return coeffs

    # Generate coeffs for half 1
    coeffs1 = generate_coeffs(N1)
    sums1 = coeffs1 @ A1
    
    # Sort sums1 and keep coeffs1 aligned
    order = np.argsort(sums1)
    sums1_sorted = sums1[order]
    coeffs1_sorted = coeffs1[order]
    
    # Generate coeffs for half 2
    coeffs2 = generate_coeffs(N2)
    sums2 = coeffs2 @ A2
    
    # Find best pair
    # For each sum2, find sum1 closest to -sum2
    # searchsorted returns index where -sum2 would be inserted
    # We check idx, idx-1, idx+1
    
    # Vectorized search
    # target = -sums2
    # idx = np.searchsorted(sums1_sorted, -sums2)
    
    # We need to handle boundaries
    idx = np.searchsorted(sums1_sorted, -sums2)
    
    # Candidate indices
    idx0 = np.clip(idx, 0, len(sums1_sorted) - 1)
    idx1 = np.clip(idx - 1, 0, len(sums1_sorted) - 1)
    idx2 = np.clip(idx + 1, 0, len(sums1_sorted) - 1)
    
    # Get corresponding sums
    s1_cand0 = sums1_sorted[idx0]
    s1_cand1 = sums1_sorted[idx1]
    s1_cand2 = sums1_sorted[idx2]
    
    # Calculate total sums
    total0 = s1_cand0 + sums2
    total1 = s1_cand1 + sums2
    total2 = s1_cand2 + sums2
    
    # Stack to find global min non-zero
    # Shape: (3 * len(sums2),)
    all_totals = np.concatenate([total0, total1, total2])
    
    # Mask out zeros
    mask = all_totals != 0
    # Replace zeros with infinity for min finding
    abs_totals = np.where(mask, np.abs(all_totals), np.inf)
    
    min_idx = np.argmin(abs_totals)
    
    if abs_totals[min_idx] == np.inf:
        # Fallback: This should not happen
        return [1] + [0] * 29

    # Recover coefficients
    # min_idx is in [0, 3 * len(sums2))
    row = min_idx // len(sums2)
    col = min_idx % len(sums2)
    
    if row == 0:
        s1_idx = idx0[col]
    elif row == 1:
        s1_idx = idx1[col]
    else:
        s1_idx = idx2[col]
        
    best_c1 = coeffs1_sorted[s1_idx]
    best_c2 = coeffs2[col]
    
    best_c = np.concatenate([best_c1, best_c2])
    
    # Convert to list of ints
    return list(best_c.astype(int))

