import numpy as np
from math import lcm

def construct():
    import time
    start_time = time.time()
    deadline = 85.0 # 5s margin
    
    # 1. LCMs
    L_A = 1
    for i in range(1, 16):
        L_A = lcm(L_A, i)
        
    L_B = 1
    for i in range(16, 31):
        L_B = lcm(L_B, i)
        
    # 2. Weights
    # wA[i] is the weight for coefficient i (1-based) in the numerator of sum_A
    # sum_A = sum(c_i * L_A / i)
    wA = [L_A // i for i in range(1, 16)]
    wB = [L_B // i for i in range(16, 31)]
    
    N_A = 3 ** 15
    N_B = 3 ** 15
    
    # Arrays to store sums (scaled) and indices
    # u = sum_A * L_B
    # v = sum_B * L_A
    u = np.empty(N_A, dtype=np.int64)
    v = np.empty(N_B, dtype=np.int64)
    idx_A = np.empty(N_A, dtype=np.int32)
    idx_B = np.empty(N_B, dtype=np.int32)
    
    # Generate A
    # We can generate sums recursively or iteratively.
    # Iterative base-3:
    # s = 0
    # for k in range(N_A):
    #   ...
    # This is O(N * 15). 1.4e7 * 15 = 2e8 operations. Might be slow in Python loop.
    
    # Better: Recursive generation using NumPy vectorization?
    # Or just a simple loop with incremental updates.
    
    # Let's try a simple loop first, it might be fast enough in PyPy/CPython if optimized.
    # Actually, 1.4e7 iterations in Python is ~1-2 seconds.
    # 15 ops per iteration is ~30 ops. Total ~4e8 ops. ~4-8 seconds.
    # This is acceptable.
    
    def gen_sums(weights, N, scale, out_arr, idx_arr):
        # weights: list of 15 ints
        # N: 3**15
        # scale: L_B or L_A
        # out_arr: np.array to fill
        # idx_arr: np.array to fill
        
        # We iterate k from 0 to N-1
        # k in base 3 gives coefficients
        # But decoding k every time is slow.
        # We can maintain the current coefficients and update incrementally.
        
        coeffs = [0] * 15
        s = 0
        out_arr[0] = 0
        idx_arr[0] = 0
        
        # We need to map base-3 index to coefficients.
        # k = c0 + c1*3 + ... + c14*3^14
        # c_i in {-1, 0, 1}.
        # Standard base 3 digits are 0, 1, 2.
        # Map 0->-1, 1->0, 2->1.
        
        # Let's just loop k and decode.
        # To speed up decoding, we can precompute powers of 3.
        powers = [1] * 15
        for i in range(1, 15):
            powers[i] = powers[i-1] * 3
            
        for k in range(1, N):
            # Update from k-1 to k
            # This is complex to do incrementally without a stack.
            # Just decode k.
            temp = k
            s = 0
            for i in range(15):
                digit = temp % 3
                temp //= 3
                # Map 0,1,2 to -1,0,1
                # 0 -> -1, 1 -> 0, 2 -> 1
                # c = digit - 1
                c = digit - 1
                s += c * weights[i]
            out_arr[k] = s * scale
            idx_arr[k] = k
            
    # This decoding loop is 1.4e7 * 15 = 2.1e8 iterations.
    # In Python, this is likely 10-20 seconds.
    # Two such loops = 20-40 seconds.
    # Sorting 1.4e7 ints: ~5-10 seconds.
    # Binary search loop: 1.4e7 * log(1.4e7) ~ 1.4e7 * 24 ~ 3.3e8 steps.
    # Numpy binary search (`searchsorted`) is vectorized!
    # `np.searchsorted` is implemented in C. It will be very fast.
    
    # So the bottleneck is the Python loop for generation.
    # Can we vectorize generation?
    # We can generate all base-3 representations using NumPy?
    # `np.arange(N_A)` -> reshape to (N_A, 15) base 3?
    # This requires 1.4e7 * 15 * 4 bytes = 840 MB. Too much?
    # Maybe.
    
    # Alternative: Split A into two halves of 7 and 8?
    # 3^7 = 2187, 3^8 = 6561.
    # Sums of 7 terms: 2187 values.
    # Sums of 8 terms: 6561 values.
    # Combine: 2187 * 6561 = 14.3M values.
    # We can generate the small lists in Python quickly.
    # Then combine them using NumPy broadcasting or outer sum.
    # `sums_7` shape (2187,), `sums_8` shape (6561,).
    # `sums_A = sums_7[:, None] + sums_8[None, :]` -> shape (2187, 6561).
    # Flatten.
    # This is much faster!
    
    def gen_sums_split(weights, N_total, scale, out_arr, idx_arr):
        # Split weights into two parts
        # Part 1: first 7 weights
        # Part 2: next 8 weights
        w1 = weights[:7]
        w2 = weights[7:]
        
        N1 = 3 ** 7
        N2 = 3 ** 8
        
        # Generate sums for part 1
        s1 = np.zeros(N1, dtype=np.int64)
        idx1 = np.zeros(N1, dtype=np.int32)
        # Generate sums for part 2
        s2 = np.zeros(N2, dtype=np.int64)
        idx2 = np.zeros(N2, dtype=np.int32)
        
        # Helper to fill small arrays
        def fill_small(weights_part, N, out_s, out_idx):
            powers = [1] * len(weights_part)
            for i in range(1, len(weights_part)):
                powers[i] = powers[i-1] * 3
            for k in range(N):
                temp = k
                s = 0
                for i in range(len(weights_part)):
                    digit = temp % 3
                    temp //= 3
                    c = digit - 1
                    s += c * weights_part[i]
                out_s[k] = s
                out_idx[k] = k
                
        fill_small(w1, N1, s1, idx1)
        fill_small(w2, N2, s2, idx2)
        
        # Combine
        # s_total = s1[:, None] + s2[None, :]
        # idx_total = idx1[:, None] * N2 + idx2[None, :]
        # This creates a 2D array of size 1.4e7.
        # Memory: 1.4e7 * 8 bytes (int64) = 112 MB for sums.
        # 1.4e7 * 4 bytes (int32) = 56 MB for indices.
        # Total ~170 MB. Acceptable.
        
        s_comb = s1[:, None] + s2[None, :]
        idx_comb = idx1[:, None] * N2 + idx2[None, :]
        
        out_arr[:] = s_comb.flatten() * scale
        idx_arr[:] = idx_comb.flatten()

    # Generate A
    gen_sums_split(wA, N_A, L_B, u, idx_A)
    
    if time.time() > deadline - 20:
         # Fallback to seed
         return [0]*28 + [1, -1]
         
    # Generate B
    gen_sums_split(wB, N_B, L_A, v, idx_B)
    
    if time.time() > deadline - 10:
         return [0]*28 + [1, -1]
         
    # Sort v
    # We need to sort v but keep track of original indices?
    # No, we just need the value v and the index idx_B corresponding to it.
    # If we sort v, we lose the association with idx_B unless we sort them together.
    # `np.argsort` returns indices.
    # Or we can create a structured array?
    # Simpler: Sort v, and also sort idx_B accordingly.
    
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    idx_B_sorted = idx_B[sort_idx]
    
    if time.time() > deadline - 5:
         return [0]*28 + [1, -1]
         
    # Binary search for each u
    # We want to find v such that u + v is min non-zero.
    # Target t = -u.
    # `pos = np.searchsorted(v_sorted, -u)`
    # Check `pos` and `pos-1`.
    
    min_diff = None
    best_i = -1
    best_j = -1
    
    # Vectorized search?
    # `pos = np.searchsorted(v_sorted, -u)` gives an array of positions.
    # Then we can check `v_sorted[pos]` and `v_sorted[pos-1]` (with bounds checks).
    
    pos = np.searchsorted(v_sorted, -u)
    
    # Handle bounds
    # pos can be 0 or N_B.
    # If pos == 0, only check 0.
    # If pos == N_B, only check N_B-1.
    
    # Candidate 1: v_sorted[pos] (if pos < N_B)
    # Candidate 2: v_sorted[pos-1] (if pos > 0)
    
    # Let's create arrays for candidates
    # We need to be careful with indices.
    
    # Instead of vectorizing the check which is tricky with bounds,
    # we can just loop?
    # 1.4e7 loop in Python is slow.
    # But `np.searchsorted` is fast.
    # The check `u[i] + v_sorted[pos[i]]` can be vectorized.
    
    # Let's try vectorizing the candidate evaluation.
    
    # Mask for valid pos < N_B
    mask1 = pos < N_B
    mask2 = pos > 0
    
    # Initialize best diff with large number
    best_diff_val = np.iinfo(np.int64).max
    best_i_vec = np.full(N_A, -1, dtype=np.int32)
    best_j_vec = np.full(N_A, -1, dtype=np.int32)
    
    # Candidate 1
    # v_cand1 = v_sorted[pos] where mask1
    # diff1 = u + v_cand1
    # We want min |diff1| != 0
    
    # To avoid out of bounds, we can use `np.take` with `mode='clip'`?
    # `np.take(v_sorted, pos, mode='clip')`
    # But if pos is out of bounds, the value is clamped, which is wrong.
    # So we must mask.
    
    # Create dummy arrays
    v_cand1 = np.zeros(N_A, dtype=np.int64)
    v_cand2 = np.zeros(N_A, dtype=np.int64)
    
    # Fill valid parts
    if np.any(mask1):
        v_cand1[mask1] = v_sorted[pos[mask1]]
    if np.any(mask2):
        v_cand2[mask2] = v_sorted[pos[mask2] - 1]
        
    diff1 = u + v_cand1
    diff2 = u + v_cand2
    
    # We want min |diff| != 0
    # abs_diff1 = np.abs(diff1)
    # abs_diff2 = np.abs(diff2)
    
    # Set invalid diffs to max
    abs_diff1 = np.where(mask1, np.abs(diff1), best_diff_val)
    abs_diff2 = np.where(mask2, np.abs(diff2), best_diff_val)
    
    # Exclude zeros
    # If diff is 0, we don't want it.
    # We can set abs_diff to max if diff == 0.
    abs_diff1 = np.where(diff1 == 0, best_diff_val, abs_diff1)
    abs_diff2 = np.where(diff2 == 0, best_diff_val, abs_diff2)
    
    # Find min for each i
    min_abs_diff = np.minimum(abs_diff1, abs_diff2)
    
    # Find global min
    global_min_val = np.min(min_abs_diff)
    
    if global_min_val == best_diff_val:
        # No non-zero sum found? Impossible.
        return [0]*28 + [1, -1]
        
    # Find index i with global min
    best_i = np.argmin(min_abs_diff)
    
    # Determine which candidate (1 or 2) was better for this i
    if abs_diff1[best_i] <= abs_diff2[best_i]:
        # Candidate 1 was better
        # j = pos[best_i]
        j_idx = pos[best_i]
    else:
        # Candidate 2 was better
        j_idx = pos[best_i] - 1
        
    # Get original indices
    i_orig = best_i
    j_orig = sort_idx[j_idx] # Map back to original B index
    
    # Get base-3 indices
    idx_a_val = idx_A[i_orig]
    idx_b_val = idx_B[j_orig]
    
    # Decode coefficients
    coeffs = [0] * 30
    
    # Decode A (first 15)
    temp = idx_a_val
    for i in range(15):
        digit = temp % 3
        temp //= 3
        coeffs[i] = digit - 1
        
    # Decode B (next 15)
    # Note: idx_B was generated with the same logic (base 3 index 0..3^15-1)
    # But the weights were for terms 16..30.
    # The index mapping is local to the B generation.
    temp = idx_b_val
    for i in range(15):
        digit = temp % 3
        temp //= 3
        coeffs[15+i] = digit - 1
        
    # Verify
    # s = sum(c/i)
    # if s == 0: return seed
    # else return coeffs
    
    return coeffs

# EVOLVE-BLOCK-END
