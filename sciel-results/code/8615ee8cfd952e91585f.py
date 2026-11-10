import numpy as np
import time
from fractions import Fraction
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # We need exactly 30 coefficients from {-1, 0, 1}
    # The signed sum S = sum(c_i / (i+1)) for i in 0..29 must be nonzero
    # We want to minimize |S|
    
    # Strategy: Meet-in-the-middle with exact arithmetic verification
    # Split into two halves of 15 terms each
    
    # Precompute weights as Fractions for exact computation
    # But for search, use float approximations and verify top candidates exactly
    
    # Generate all 3^15 = 14348907 possibilities for each half
    # This is feasible with numpy
    
    n_half = 15
    total_combos = 3 ** n_half  # 14348907
    
    # Generate coefficients for first half (terms 1-15)
    # Each coefficient is in {-1, 0, 1}
    # We'll use base-3 representation
    
    # For memory efficiency, we'll process in chunks or use a smarter approach
    # Actually, 3^15 = 14348907 is manageable
    
    # Let's generate all combinations and their sums
    # Term weights for first half: 1/1, 1/2, ..., 1/15
    # Term weights for second half: 1/16, 1/17, ..., 1/30
    
    weights_first = np.array([1.0 / (i + 1) for i in range(n_half)])
    weights_second = np.array([1.0 / (i + 1) for i in range(n_half, 2 * n_half)])
    
    # Generate all 3^15 coefficient vectors and their weighted sums
    # Use iterative generation to save memory
    
    def gen_half_sums(weights):
        """Generate all possible sums for a half with given weights."""
        n = len(weights)
        # Start with one sum: 0.0
        sums = np.array([0.0])
        coeffs = np.zeros((1, n), dtype=np.int8)
        
        for i in range(n):
            w = weights[i]
            current_size = len(sums)
            
            # Create new arrays of size 3 * current_size
            new_size = current_size * 3
            new_sums = np.empty(new_size)
            new_coeffs = np.empty((new_size, n), dtype=np.int8)
            
            # Coeff 0
            new_sums[:current_size] = sums
            new_coeffs[:current_size] = coeffs
            
            # Coeff 1
            new_sums[current_size:2*current_size] = sums + w
            new_coeffs[current_size:2*current_size] = coeffs
            new_coeffs[current_size:2*current_size, i] = 1
            
            # Coeff -1
            new_sums[2*current_size:] = sums - w
            new_coeffs[2*current_size:] = coeffs
            new_coeffs[2*current_size:, i] = -1
            
            sums = new_sums
            coeffs = new_coeffs
            
        return sums, coeffs
    
    # Generate first half
    sums_A, coeffs_A = gen_half_sums(weights_first)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Generate second half
    sums_B, coeffs_B = gen_half_sums(weights_second)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Sort sums_B for binary search
    idx_sort = np.argsort(sums_B)
    sums_B_sorted = sums_B[idx_sort]
    coeffs_B_sorted = coeffs_B[idx_sort]
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # For each sum in A, find the sum in B that minimizes |sum_A + sum_B|
    # This is equivalent to finding sum_B closest to -sum_A
    
    targets = -sums_A
    
    # Use searchsorted to find insertion points
    len_B = len(sums_B_sorted)
    idx_B = np.searchsorted(sums_B_sorted, targets)
    
    # Check both the insertion point and the element before it
    idx_B_clipped_0 = np.clip(idx_B, 0, len_B - 1)
    idx_B_clipped_1 = np.clip(idx_B - 1, 0, len_B - 1)
    
    # Calculate candidate sums
    cand_sums_0 = sums_A + sums_B_sorted[idx_B_clipped_0]
    cand_sums_1 = sums_A + sums_B_sorted[idx_B_clipped_1]
    
    # Combine into 2D array
    cand_sums = np.column_stack((cand_sums_0, cand_sums_1))
    abs_cands = np.abs(cand_sums)
    
    # Mask out values that are exactly zero (we need nonzero sum)
    zero_mask = abs_cands < 1e-18
    abs_cands[zero_mask] = np.inf
    
    # Find top K candidates to verify exactly
    K = 200
    flat_indices = np.argpartition(abs_cands.ravel(), K)[:K]
    
    best_coeff = None
    best_val = Fraction(1, 1)  # Start with 1 as upper bound
    
    # Precompute for exact verification
    for idx in flat_indices:
        row = idx // 2
        col = idx % 2
        
        # Get coefficients
        best_a = coeffs_A[row]
        
        if col == 0:
            b_idx = idx_B_clipped_0[row]
        else:
            b_idx = idx_B_clipped_1[row]
            
        best_b = coeffs_B_sorted[b_idx]
        
        # Construct full coefficient vector
        full_coeffs = np.concatenate([best_a, best_b])
        
        # Calculate exact sum
        s = Fraction(0, 1)
        for i in range(30):
            c = int(full_coeffs[i])
            if c != 0:
                s += Fraction(c, i + 1)
        
        if s == 0:
            continue
            
        abs_s = abs(s)
        
        if abs_s < best_val:
            best_val = abs_s
            best_coeff = full_coeffs.copy()
        
        if time.time() > deadline:
            break
    
    if best_coeff is None:
        # Fallback: return a simple nonzero sum
        # c_0 = 1 gives sum = 1
        fallback = [1] + [0]*29
        return fallback
        
    return [int(x) for x in best_coeff]
