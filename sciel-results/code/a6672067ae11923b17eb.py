import numpy as np

def construct():
    N = 10000000
    target_ratio_num = 3
    target_ratio_den = 2
    
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Sort indices by sigma value
    # This is O(N log N) but in C
    order = np.argsort(sigma, kind='stable')
    sorted_sigma = sigma[order]
    
    # Find boundaries of groups with same sigma
    # diff = 0 where sigma is same
    # We want indices where sorted_sigma[i] != sorted_sigma[i-1]
    # np.diff returns length N-1
    # boundaries are 0 and indices where diff != 0 + 1
    diff = np.diff(sorted_sigma)
    change_indices = np.nonzero(diff)[0] + 1
    boundaries = np.concatenate(([0], change_indices, [N]))
    
    best_err_num = 10**18 # Large numerator for error
    best_err_den = 1
    best_a = 1
    best_b = 1
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        if end - start < 2:
            continue
            
        # Get the numbers in this group
        # order[start:end] contains the indices n
        nums = order[start:end]
        # Sort them numerically
        nums = np.sort(nums)
        
        # Iterate through each b in this group
        # We can vectorize the target calculation and search?
        # np.searchsorted is vectorized!
        targets = 1.5 * nums
        # But we need to compare exact rationals, so we must do the final check in Python or vectorize the rational comparison.
        # Rational comparison: |a/b - 3/2| = |2a - 3b| / 2b
        # We want to minimize |2a - 3b| / (2b)
        # Let err_num = |2a - 3b|, err_den = 2b
        # Compare err_num1/err_den1 < err_num2/err_den2 => err_num1 * err_den2 < err_num2 * err_den1
        
        # First, find candidate a's using float searchsorted
        # pos = np.searchsorted(nums, targets)
        # This gives the index where target would be inserted.
        # Candidates are pos and pos-1.
        
        pos = np.searchsorted(nums, targets)
        
        # We need to check pos and pos-1 for each b.
        # pos can be out of bounds.
        # Let's create arrays for candidates.
        
        # Candidate 1: nums[pos]
        # Clip pos to [0, len(nums)-1]
        pos_clipped = np.clip(pos, 0, len(nums) - 1)
        cand_a1 = nums[pos_clipped]
        
        # Candidate 2: nums[pos-1]
        # If pos is 0, pos-1 is -1. We can handle this by checking bounds.
        pos_m1 = pos - 1
        valid_m1 = pos_m1 >= 0
        pos_m1_clipped = np.clip(pos_m1, 0, len(nums) - 1)
        cand_a2 = nums[pos_m1_clipped]
        
        # We need to ensure a != b.
        # Also, we need to compute the exact error for these candidates.
        
        # Let's process in chunks or just iterate?
        # The group sizes can be large.
        # But total size is N.
        # Iterating in Python over all N elements is what the current program does (roughly).
        # Can we vectorize the error calculation?
        
        # err1_num = |2 * cand_a1 - 3 * nums|
        # err1_den = 2 * nums
        # err2_num = |2 * cand_a2 - 3 * nums|
        # err2_den = 2 * nums
        
        # We want the min of err1 and err2 (where valid).
        
        # Let's just do a Python loop over the group if it's small, or vectorize if large?
        # Actually, a Python loop over 10^7 items is ~1s.
        # The current program does this.
        # The improvement is:
        # 1. No defaultdict overhead.
        # 2. Exact rational comparison.
        # 3. Vectorized searchsorted to get candidates.
        
        # Let's try to vectorize the final comparison.
        
        # Calculate errors for cand_a1
        e1_num = np.abs(2 * cand_a1 - 3 * nums)
        e1_den = 2 * nums
        
        # Calculate errors for cand_a2 (only where valid_m1)
        e2_num = np.abs(2 * cand_a2 - 3 * nums)
        e2_den = 2 * nums
        
        # Mask out invalid candidates
        # For cand_a1, it's always valid index-wise, but might be a==b.
        # For cand_a2, valid only if pos > 0.
        
        # We want to find the min of e1 and e2.
        # Compare e1 and e2: e1_num * e2_den < e2_num * e1_den
        # Note: e1_den == e2_den == 2 * nums.
        # So we just need to compare e1_num and e2_num!
        # Because the denominator is the same for both candidates for a given b.
        
        # So for each b, the best a is the one with smaller |2a - 3b|.
        
        # Let's create a mask for which candidate is better.
        # valid1 = np.ones(len(nums), dtype=bool)
        # valid2 = valid_m1
        
        # If valid2, compare e1_num and e2_num.
        # best_num = np.where(valid2 & (e2_num < e1_num), e2_num, e1_num)
        # best_a = np.where(valid2 & (e2_num < e1_num), cand_a2, cand_a1)
        
        # But we also need to check a != b.
        # If best_a == b, we must pick the other one (if valid) or skip.
        
        # This is getting complex to vectorize fully with the a!=b constraint.
        # Let's stick to a Python loop but with pre-computed candidates from NumPy.
        
        # Actually, we can just iterate. The NumPy part saved the sorting and grouping.
        
        for idx in range(len(nums)):
            b = nums[idx]
            
            # Get candidates from pre-computed arrays
            a1 = cand_a1[idx]
            a2 = cand_a2[idx] if valid_m1[idx] else -1
            
            # Determine best a among a1, a2
            # Error num for a1: |2a1 - 3b|
            # Error num for a2: |2a2 - 3b|
            
            e1 = abs(2 * a1 - 3 * b)
            e2 = abs(2 * a2 - 3 * b) if a2 != -1 else 10**18
            
            if e1 < e2:
                best_a_local = a1
                best_e_local = e1
            else:
                best_a_local = a2
                best_e_local = e2
                
            # Check a != b
            if best_a_local == b:
                # Pick the other one
                if a1 == b and a2 != -1:
                    best_a_local = a2
                    best_e_local = e2
                elif a2 == b and a1 != -1: # a1 is never -1
                    best_a_local = a1
                    best_e_local = e1
                else:
                    continue # No valid a
            
            # Compare with global best
            # Global error: best_e_local / (2 * b)
            # Current best: best_err_num / best_err_den
            # Compare: best_e_local * best_err_den < best_err_num * (2 * b)
            
            if best_e_local * best_err_den < best_err_num * (2 * b):
                best_err_num = best_e_local
                best_err_den = 2 * b
                best_a = best_a_local
                best_b = b
                
    # Ensure result is not exactly 0
    if best_a == 0 or best_b == 0:
        return [1, 1]
        
    return [int(best_a), int(best_b)]
