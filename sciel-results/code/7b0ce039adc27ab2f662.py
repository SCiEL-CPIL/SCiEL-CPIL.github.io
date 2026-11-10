import numpy as np
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    n = 30
    half = 15
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute scaled terms: term[i] = L / (i+1) for i in 0..29
    # term[i] corresponds to coefficient of 1/(i+1)
    terms = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # 2. Generate all sums for first half (indices 0 to 14)
    # There are 3^15 = 14,348,907 combinations
    # We represent a combination by an index in base 3.
    # coeff[i] is the digit at position i in base 3.
    # Value = sum(coeff[i] * terms[i])
    
    size = 3 ** half
    
    # To save memory and time, we generate the sums for the first half.
    # We will store the sums and the index (to recover coefficients later if needed, 
    # but we can just store the index directly).
    # Actually, we can generate the sums on the fly or in a loop.
    
    # Efficient generation using recursion or iterative base-3 counting?
    # Iterative base-3 counting is simple.
    
    # Let's generate sums_A
    sums_A = np.empty(size, dtype=np.int64)
    # We also need to recover the coefficients for the best candidate.
    # We can store the index for each sum.
    # But multiple indices can have the same sum. We just need one index for the best sum.
    # So we can just store the index that generated the sum?
    # No, we iterate through all sums_A. For each sum_A, we find best sum_B.
    # We need to know which index in A and which index in B gave the best result.
    
    # Let's generate sums_A and store the index that produced them?
    # It's easier to just iterate through indices 0 to size-1, compute the sum,
    # and keep the index if it's the best.
    
    # But we need to sort sums_B.
    
    # Step 2a: Generate sums_B and sort them
    sums_B = np.empty(size, dtype=np.int64)
    # We need to map back from sorted sums_B to the original index to get coefficients.
    # So we store indices_B = np.arange(size)
    # Then sort sums_B and indices_B together.
    
    idx_B = np.arange(size, dtype=np.int64)
    
    # Fill sums_B
    # terms_B = terms[15:30]
    terms_B = terms[15:30]
    
    # We can fill sums_B by iterating through base-3 numbers.
    # Or using a more vectorized approach?
    # For 14M elements, a Python loop might be slow (14M iterations).
    # Let's try to vectorize or use a faster method.
    
    # A recursive generation might be faster if done in C, but we are in Python.
    # Let's use a loop but optimize it.
    
    # Actually, we can generate the sums using a recursive function that fills the array.
    # Or we can use the fact that the sums are linear combinations.
    
    # Let's try a simple loop first and see if it's fast enough.
    # 14M iterations in Python might take ~1-2 seconds. That's acceptable.
    
    def gen_sums(terms_part, size):
        sums = np.empty(size, dtype=np.int64)
        # Base-3 digits
        # We can compute the sum for each index i by converting i to base 3.
        # This is O(log_3(size)) per element, which is O(15) operations.
        # 14M * 15 = 210M operations. This might be too slow in pure Python.
        
        # Alternative: Build up the array recursively.
        # sums = [0]
        # for t in terms_part:
        #   sums = sums + np.concatenate([sums - t, sums, sums + t])
        # This is O(N) per step, total O(N * 15).
        # N grows by factor 3 each step.
        # Step 1: 3
        # Step 2: 9
        # ...
        # Step 15: 14M
        # Total operations: sum(3^k * 1) for k=1..15? No, the concatenation is O(current_size).
        # Sum of sizes: 3 + 9 + ... + 3^15 = (3^16 - 3)/2 ~ 21M.
        # This is much faster!
        
        current_sums = np.array([0], dtype=np.int64)
        for t in terms_part:
            # new_sums = [s - t, s, s + t for s in current_sums]
            # We can do this with np.concatenate
            # Order doesn't matter for sorting, but we need to keep track of indices if we were storing them.
            # Since we are just generating sums, we don't need indices here yet.
            # But we need to map the final sorted sums back to coefficients.
            # If we just generate sums, we lose the coefficient info.
            # We need to recover coefficients from the index.
            # So we MUST use the index-based approach or store the path.
            
            # If we use the index-based approach (base 3), we can recover coefficients from the index.
            # So we just need the sums.
            pass
            
        # Let's stick to the index-based approach for recovery, but we need to compute sums.
        # Can we compute sums for all indices in a vectorized way?
        # Yes.
        # Let indices = np.arange(size)
        # Digits = indices // powers_of_3 % 3
        # Coeffs = 2 * Digits - 1  (maps 0->-1, 1->0, 2->1)
        # Sums = Coeffs @ terms_part
        
        powers = 3 ** np.arange(half)
        indices = np.arange(size, dtype=np.int64)
        # This creates an array of shape (size, half) which is 14M * 15 = 210M ints.
        # 210M * 8 bytes = 1.6 GB. This might exceed memory.
        
        # We need a more memory-efficient way to compute sums.
        # We can compute sums in chunks or use the recursive generation and store indices.
        
        # Let's use the recursive generation but store the index for each sum.
        # This doubles memory for the array (sums + indices).
        # 14M * 8 bytes (sums) + 14M * 8 bytes (indices) = 224 MB. This is fine.
        
        sums_arr = np.array([0], dtype=np.int64)
        idx_arr = np.array([0], dtype=np.int64)
        
        for i, t in enumerate(terms_part):
            # Current size is len(sums_arr)
            # New size is 3 * len(sums_arr)
            # New sums:
            # 1. coeff -1: sums_arr - t
            # 2. coeff 0: sums_arr
            # 3. coeff 1: sums_arr + t
            
            # New indices:
            # The new digit at position i is -1, 0, or 1.
            # But our index is in base 3, where digits are 0, 1, 2.
            # Let's map:
            # Digit 0 -> coeff -1
            # Digit 1 -> coeff 0
            # Digit 2 -> coeff 1
            # So if we add a new digit d (0,1,2) at position i, the new index is:
            # old_index * 3 + d ? No, this depends on endianness.
            # Let's say index is represented as d_0 d_1 ... d_{k-1} in base 3.
            # Value = sum d_j * 3^j.
            # If we append a new digit at the most significant position (position k):
            # new_index = d_k * 3^k + old_index.
            # Let's use this convention.
            
            size_curr = len(sums_arr)
            new_sums = np.empty(size_curr * 3, dtype=np.int64)
            new_idx = np.empty(size_curr * 3, dtype=np.int64)
            
            # Digit 0 (coeff -1)
            new_sums[0:size_curr] = sums_arr - t
            new_idx[0:size_curr] = idx_arr + 0 * (3 ** i) # Wait, 3**i is the place value.
            # If we are adding digit at position i, and we have processed i digits so far,
            # the previous indices are formed by digits 0..i-1.
            # So new_index = old_index + d * 3^i.
            
            # Digit 1 (coeff 0)
            new_sums[size_curr:2*size_curr] = sums_arr
            new_idx[size_curr:2*size_curr] = idx_arr + 1 * (3 ** i)
            
            # Digit 2 (coeff 1)
            new_sums[2*size_curr:3*size_curr] = sums_arr + t
            new_idx[2*size_curr:3*size_curr] = idx_arr + 2 * (3 ** i)
            
            sums_arr = new_sums
            idx_arr = new_idx
            
        return sums_arr, idx_arr

    # Generate for A (terms 0-14)
    sums_A, idx_A = gen_sums(terms[0:15], size)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Generate for B (terms 15-29)
    sums_B, idx_B = gen_sums(terms[15:30], size)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]

    # Sort sums_B and reorder idx_B
    sort_idx = np.argsort(sums_B)
    sums_B_sorted = sums_B[sort_idx]
    idx_B_sorted = idx_B[sort_idx]
    
    # Now search
    best_val = None
    best_idx_A = -1
    best_idx_B = -1
    
    # For each sum in A, find closest in B to -sum_A
    # We can vectorize the searchsorted part.
    targets = -sums_A
    
    # searchsorted returns the index where the target would be inserted
    pos = np.searchsorted(sums_B_sorted, targets)
    
    # Clip to valid range
    pos_clipped_0 = np.clip(pos, 0, size - 1)
    pos_clipped_1 = np.clip(pos - 1, 0, size - 1)
    
    # Get the candidate sums from B
    cand_sums_B_0 = sums_B_sorted[pos_clipped_0]
    cand_sums_B_1 = sums_B_sorted[pos_clipped_1]
    
    # Calculate total sums
    total_sums_0 = sums_A + cand_sums_B_0
    total_sums_1 = sums_A + cand_sums_B_1
    
    # We want the minimum non-zero absolute value
    # abs_total_0 = np.abs(total_sums_0)
    # abs_total_1 = np.abs(total_sums_1)
    
    # Mask zeros
    # We need to find the global minimum.
    # Let's combine into one array? Or just find min of each and compare.
    
    # To avoid huge memory, we can find the min of each array separately.
    
    # But we need to exclude 0.
    # If min is 0, we need the next smallest.
    
    # Let's find the indices of the smallest values.
    # Since we only care about the global minimum, we can find the min of abs(total_sums_0) 
    # that is not 0, and same for 1.
    
    # However, there might be many zeros.
    # A safer way:
    # Create a mask for non-zero.
    # But creating a 14M boolean array is fine.
    
    mask_0 = total_sums_0 != 0
    mask_1 = total_sums_1 != 0
    
    # If all are zero, this will fail, but that's unlikely.
    
    # To find the min absolute value among non-zeros:
    # We can set zeros to inf and take min.
    
    abs_total_0 = np.abs(total_sums_0)
    abs_total_1 = np.abs(total_sums_1)
    
    abs_total_0[~mask_0] = np.iinfo(np.int64).max
    abs_total_1[~mask_1] = np.iinfo(np.int64).max
    
    min_val_0 = np.min(abs_total_0)
    min_idx_0 = np.argmin(abs_total_0)
    
    min_val_1 = np.min(abs_total_1)
    min_idx_1 = np.argmin(abs_total_1)
    
    if min_val_0 < min_val_1:
        best_val = min_val_0
        best_idx_A = min_idx_0
        best_idx_B = pos_clipped_0[min_idx_0]
    else:
        best_val = min_val_1
        best_idx_A = min_idx_1
        best_idx_B = pos_clipped_1[min_idx_1]
        
    if best_val is None or best_val == np.iinfo(np.int64).max:
        # Fallback
        return [0]*28 + [1, -1]
        
    # Recover coefficients
    # idx_A is the base-3 index for the first half
    # idx_B_sorted[best_idx_B] is the base-3 index for the second half
    
    coeff_idx_A = idx_A[best_idx_A]
    coeff_idx_B = idx_B_sorted[best_idx_B]
    
    coeffs = np.zeros(30, dtype=int)
    
    # Decode base-3
    # Digit i corresponds to term i (0-indexed)
    # Value = d * 3^i
    # We need to extract d_i = (index // 3^i) % 3
    # Then map: 0->-1, 1->0, 2->1
    
    powers = 3 ** np.arange(15)
    
    # For A
    digits_A = (coeff_idx_A // powers) % 3
    # Map to -1, 0, 1
    # d=0 -> -1, d=1 -> 0, d=2 -> 1
    # coeffs[i] = 2 * d - 1 ? No.
    # 0 -> -1
    # 1 -> 0
    # 2 -> 1
    # Formula: if d==0: -1, elif d==1: 0, else: 1
    # Or: 2*d - 1 works for d=0->-1, d=1->1 (wrong), d=2->3 (wrong).
    # Let's just use a lookup or simple logic.
    # coeffs_A = np.where(digits_A == 0, -1, np.where(digits_A == 1, 0, 1))
    
    coeffs[0:15] = np.where(digits_A == 0, -1, np.where(digits_A == 1, 0, 1))
    
    # For B
    digits_B = (coeff_idx_B // powers) % 3
    coeffs[15:30] = np.where(digits_B == 0, -1, np.where(digits_B == 1, 0, 1))
    
    # Verify
    # s = sum(coeffs[i] / (i+1))
    # It should be non-zero.
    
    return [int(x) for x in coeffs]
