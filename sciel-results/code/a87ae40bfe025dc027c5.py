import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Weights for first 15 terms (1..15)
    W_A = np.array([L // i for i in range(1, 16)], dtype=np.int64)
    # Weights for next 15 terms (16..30)
    W_B = np.array([L // i for i in range(16, 31)], dtype=np.int64)
    
    # 2. Generate all sums for A
    # Number of combinations: 3^15
    n_combos = 3**15
    
    # Generate indices 0 to 3^15 - 1
    # We need to decode these into base-3 digits to get coefficients
    # To avoid huge memory usage for intermediate arrays, we can do this in chunks or just use the full array if memory allows.
    # 14,348,907 * 15 * 8 bytes (int64) ~ 1.7 GB. This is likely too high for some CI environments.
    # Let's process in chunks of 1 million.
    
    chunk_size = 1_000_000
    all_sums_A = []
    all_indices_A = []
    
    # Powers of 3 for decoding
    powers_3 = np.array([3**i for i in range(15)], dtype=np.int64)
    
    for start in range(0, n_combos, chunk_size):
        end = min(start + chunk_size, n_combos)
        # Generate indices for this chunk
        idx_chunk = np.arange(start, end, dtype=np.int64)
        
        # Decode to base 3
        # digits = (idx_chunk[:, None] // powers_3) % 3
        # This creates a (chunk_size, 15) array
        digits = (idx_chunk[:, None] // powers_3) % 3
        
        # Convert digits (0,1,2) to coeffs (-1,0,1)
        coeffs = digits - 1
        
        # Calculate sums
        sums = coeffs @ W_A
        
        all_sums_A.append(sums)
        all_indices_A.append(idx_chunk)
        
        if time.time() - start_time > deadline - 10:
            break
            
    if not all_sums_A:
        # Fallback if timeout during generation (unlikely)
        return [0]*28 + [1, -1]
        
    sums_A = np.concatenate(all_sums_A)
    indices_A = np.concatenate(all_indices_A)
    
    # 3. Create mapping from Sum to Index for A
    # We want to keep one index for each unique sum.
    # np.unique returns sorted unique values and indices of first occurrence
    unique_sums_A, unique_indices_in_A = np.unique(sums_A, return_index=True)
    
    # Map from sum value to original index
    # We can use a dictionary for O(1) lookup later if needed, 
    # but since we will use searchsorted on unique_sums_A, we just need 
    # a way to map the unique sum back to an index.
    # unique_indices_in_A gives the position in the original arrays.
    # We need the actual index value (the integer 0..3^15-1) to decode coefficients later.
    # indices_A[unique_indices_in_A] gives the actual indices.
    actual_indices_for_unique_sums = indices_A[unique_indices_in_A]
    
    # 4. Generate all sums for B
    all_sums_B = []
    all_indices_B = []
    
    for start in range(0, n_combos, chunk_size):
        end = min(start + chunk_size, n_combos)
        idx_chunk = np.arange(start, end, dtype=np.int64)
        
        digits = (idx_chunk[:, None] // powers_3) % 3
        coeffs = digits - 1
        sums = coeffs @ W_B
        
        all_sums_B.append(sums)
        all_indices_B.append(idx_chunk)
        
        if time.time() - start_time > deadline - 5:
            break
            
    if not all_sums_B:
        # Fallback
        return [0]*28 + [1, -1]
        
    sums_B = np.concatenate(all_sums_B)
    indices_B = np.concatenate(all_indices_B)
    
    # 5. Find best match
    # For each s_B, we want s_A such that |s_A + s_B| is min and non-zero.
    # Let target = -s_B. We want s_A closest to target.
    
    # searchsorted returns the index where target would be inserted to maintain order.
    # Let pos = searchsorted(unique_sums_A, -sums_B)
    # Candidates are unique_sums_A[pos] and unique_sums_A[pos-1]
    
    # Be careful with boundaries: pos can be 0 or len(unique_sums_A)
    
    pos = np.searchsorted(unique_sums_A, -sums_B)
    
    # Initialize best difference to infinity
    best_diff = np.full(len(sums_B), np.iinfo(np.int64).max, dtype=np.int64)
    best_sum_A_idx = np.zeros(len(sums_B), dtype=np.int64) # Index into unique_sums_A
    
    # Check candidate at pos
    # valid_pos = (pos >= 0) & (pos < len(unique_sums_A))
    # Since searchsorted returns 0..N, and we check pos and pos-1, 
    # pos is always in [0, N].
    
    # Candidate 1: at index pos (if pos < N)
    valid_pos1 = pos < len(unique_sums_A)
    if np.any(valid_pos1):
        s_A_cand1 = unique_sums_A[pos[valid_pos1]]
        diff1 = np.abs(s_A_cand1 + sums_B[valid_pos1])
        # Exclude zero
        valid_nonzero1 = diff1 > 0
        if np.any(valid_nonzero1):
            # Update best_diff and best_sum_A_idx
            curr_best = best_diff[valid_pos1]
            update_mask = valid_nonzero1 & (diff1 < curr_best)
            
            # This logic is getting complicated with vectorized masking on sub-arrays.
            # Let's simplify by padding or using full arrays with safe indexing.
            pass
            
    # Let's use a safer vectorized approach:
    # Create arrays for candidate indices, clamped to [0, N-1]
    N_A = len(unique_sums_A)
    
    pos_clamped = np.clip(pos, 0, N_A - 1)
    pos_minus1_clamped = np.clip(pos - 1, 0, N_A - 1)
    
    # Get candidate sums
    s_A_cand1 = unique_sums_A[pos_clamped]
    s_A_cand2 = unique_sums_A[pos_minus1_clamped]
    
    # Calculate differences
    diff1 = np.abs(s_A_cand1 + sums_B)
    diff2 = np.abs(s_A_cand2 + sums_B)
    
    # Zero out differences where sum is 0
    diff1[diff1 == 0] = np.iinfo(np.int64).max
    diff2[diff2 == 0] = np.iinfo(np.int64).max
    
    # Determine which candidate is better
    # We want min(diff1, diff2)
    # But we also need to know which one it was to retrieve the index.
    
    use_cand1 = diff1 <= diff2
    
    best_diff = np.where(use_cand1, diff1, diff2)
    best_pos = np.where(use_cand1, pos_clamped, pos_minus1_clamped)
    
    # Find the global minimum
    min_diff = np.min(best_diff)
    
    if min_diff == np.iinfo(np.int64).max:
        # No non-zero sum found? Impossible.
        return [0]*28 + [1, -1]
        
    # Find one index that achieves this minimum
    min_indices = np.where(best_diff == min_diff)[0]
    if len(min_indices) == 0:
        return [0]*28 + [1, -1]
        
    chosen_B_idx = min_indices[0]
    chosen_sum_B = sums_B[chosen_B_idx]
    chosen_A_pos = best_pos[chosen_B_idx]
    chosen_sum_A = unique_sums_A[chosen_A_pos]
    
    # Verify
    total_N = chosen_sum_A + chosen_sum_B
    if total_N == 0:
        # Fallback to next best? 
        # The logic above should have excluded 0.
        # If this happens, it's a bug. Let's pick the next min.
        # For now, just return a valid default.
        return [0]*28 + [1, -1]
        
    # Retrieve the actual indices for A and B
    # For A: We need the index corresponding to chosen_sum_A.
    # We have actual_indices_for_unique_sums, which maps unique_sums_A[i] -> original index.
    idx_A = actual_indices_for_unique_sums[chosen_A_pos]
    
    # For B: We have the index in the generated array.
    idx_B = indices_B[chosen_B_idx]
    
    # Decode coefficients
    coeffs = [0] * 30
    
    # Decode A (first 15)
    idx = idx_A
    for i in range(14, -1, -1):
        d = idx % 3
        coeffs[i] = d - 1
        idx //= 3
        
    # Decode B (last 15)
    idx = idx_B
    for i in range(29, 14, -1):
        d = idx % 3
        coeffs[i] = d - 1
        idx //= 3
        
    return coeffs

