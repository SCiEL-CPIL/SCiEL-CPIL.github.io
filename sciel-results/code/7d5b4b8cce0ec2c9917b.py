"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that sum(c[i]/(i+1)) is nonzero
    and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # We represent the sum as an integer S = sum(c[i] * L / (i+1))
    # We want to minimize |S| for S != 0.
    # The actual value is S / L.
    
    # Precompute weights
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        Returns a numpy array of sums.
        """
        # Number of combinations is 3^len(weights_half)
        # We can generate them iteratively.
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        
        for w in weights_half:
            # For each existing sum, we can add -w, 0, or +w
            # sums_new = [s-w for s in sums] + [s for s in sums] + [s+w for s in sums]
            # This triples the size.
            sums = np.concatenate([sums - w, sums, sums + w])
            
        return sums

    # Generate sums for both halves
    # 3^15 is ~14.3 million. This might use significant memory (14.3M * 8 bytes * 3 arrays during concat ~ 350MB peak?)
    # Let's be careful. 
    # 14.3M int64 is ~115MB.
    # Concatenating creates new arrays.
    # It should be fine on most evaluators.
    
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    best_abs_S = np.iinfo(np.int64).max
    best_coeff = None
    
    # We need to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Iterate through sums1.
    # To speed up, we can iterate through unique sums1 if there are duplicates, but there are likely few.
    # However, 14.3M iterations in Python loop is slow.
    # We can vectorize the search?
    # np.searchsorted can take an array of values.
    
    # Let's try vectorized search.
    # For each s1 in sums1, we want to find s2 in sums2_sorted close to -s1.
    # target = -sums1
    # indices = np.searchsorted(sums2_sorted, -sums1)
    
    # But 14.3M elements in sums1.
    # searchsorted is O(N log M). 14.3M * log(14.3M) ~ 14.3M * 24 ~ 340M ops.
    # NumPy handles this fast.
    
    # However, we need to check neighbors.
    # Let's do it in chunks to avoid memory issues or just let NumPy do it.
    
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)
    
    # We need to check indices and indices-1 (and maybe indices+1 if index is at boundary or value matches exactly?)
    # searchsorted returns the index where the value would be inserted to maintain order.
    # If the value exists, it returns the leftmost index.
    # So we should check:
    # 1. The element at `indices` (if valid)
    # 2. The element at `indices - 1` (if valid)
    
    # We also need to handle the case where s1 + s2 == 0. We want the smallest NON-ZERO.
    # So if s1 + s2 == 0, we should look for the next closest non-zero.
    # This complicates the vectorized approach slightly.
    
    # Let's calculate candidates:
    # cand_idx_right = indices
    # cand_idx_left = indices - 1
    
    # We must clip indices to valid range [0, len(sums2_sorted)-1]
    
    len_s2 = len(sums2_sorted)
    
    # Right candidates
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    
    # Left candidates
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    
    # We have two candidate totals for each s1: total_r and total_l.
    # We want the min |total| for total != 0.
    
    # Mask out zeros
    mask_r = total_r != 0
    mask_l = total_l != 0
    
    # If both are zero, we need to look further?
    # If s1 + s2 == 0, it means there is a zero sum.
    # The next best is the closest non-zero.
    # This requires checking idx_r + 1 or idx_l - 1 if the primary candidate is 0.
    
    # Let's create an array of all candidate totals and their corresponding coefficients?
    # No, we just need the value first. Then we can reconstruct.
    
    # Let's combine candidates.
    # To be safe and simple, let's just evaluate the min absolute value among non-zero candidates.
    
    abs_total_r = np.abs(total_r)
    abs_total_l = np.abs(total_l)
    
    # Set zero candidates to infinity
    abs_total_r[~mask_r] = np.iinfo(np.int64).max
    abs_total_l[~mask_l] = np.iinfo(np.int64).max
    
    # Find the global minimum
    min_abs_r = np.min(abs_total_r)
    min_abs_l = np.min(abs_total_l)
    
    # Determine which candidate is best
    if min_abs_r < min_abs_l:
        best_abs_S = min_abs_r
        idx_best = np.argmin(abs_total_r)
        s1_val = sums1[idx_best]
        s2_val = s2_r[idx_best]
        side = 'r'
    else:
        best_abs_S = min_abs_l
        idx_best = np.argmin(abs_total_l)
        s1_val = sums1[idx_best]
        s2_val = s2_l[idx_best]
        side = 'l'
        
    # If best_abs_S is still inf, we might have missed the closest non-zero due to the zero trap.
    # Let's do a small loop around the best candidate to ensure we find the true min non-zero.
    # We know the target s1_val.
    # Let's find all s2 in sums2_sorted close to -s1_val.
    
    target = -s1_val
    # Find indices of target in sums2_sorted
    # Since sums2_sorted might have duplicates, we want the range.
    left_idx = np.searchsorted(sums2_sorted, target, side='left')
    right_idx = np.searchsorted(sums2_sorted, target, side='right')
    
    # Check neighbors in a window around [left_idx, right_idx]
    # The closest non-zero sum will be at the boundaries of the zero-sum block or just outside.
    # Let's scan a small window around the insertion point.
    
    # We already have s1_val.
    # Let's just iterate s2 candidates from the sorted list around the target.
    
    # Get a window of indices
    # We want the smallest |s1 + s2| != 0.
    # This is equivalent to smallest |s2 - (-s1)| != 0.
    
    # Let's just check a range of indices around the insertion point.
    # The number of duplicates is likely small.
    
    start_scan = max(0, left_idx - 10)
    end_scan = min(len_s2, right_idx + 10)
    
    # This window might miss the true minimum if the zero-sum block is large and the next non-zero is far.
    # However, for random-like weights, duplicates are rare.
    # If best_abs_S is still very large (inf), we need a fallback.
    
    # Actually, the vectorized search over ALL s1 should have found the global min non-zero,
    # UNLESS the global min non-zero pair (s1*, s2*) was not a neighbor in the searchsorted sense
    # for s1*. But searchsorted finds the closest element. The closest element to -s1* is s2*.
    # So if s1*+s2* != 0, it would be found.
    # The only issue is if the *true* global min non-zero sum S* is formed by s1* + s2*,
    # but s2* is NOT the closest element to -s1* in sums2_sorted because there is a zero-sum element s2_zero
    # closer to -s1*?
    # If s1* + s2_zero = 0, then s2_zero = -s1*.
    # If s2* is the closest non-zero, and s2_zero is closer, then |s2_zero - (-s1*)| < |s2* - (-s1*)|.
    # This implies 0 < |s1* + s2*|.
    # The searchsorted would find s2_zero first.
    # We mask out zeros. So we ignore s2_zero.
    # But we only check idx and idx-1.
    # If s2_zero is at idx, and s2* is at idx+1 (or idx-1), we might miss it if we only check idx and idx-1.
    # Specifically, if s2_zero is at idx, idx points to the first zero.
    # We check idx (zero, masked) and idx-1.
    # If s2* is at idx+1, we miss it!
    
    # So we need to check idx+1 as well, or handle the zero block properly.
    
    # Let's refine the candidate generation.
    # For each s1, check idx-1, idx, idx+1.
    
    idx_m = np.clip(indices - 1, 0, len_s2 - 1)
    idx_p = np.clip(indices + 1, 0, len_s2 - 1)
    
    s2_m = sums2_sorted[idx_m]
    s2_p = sums2_sorted[idx_p]
    
    total_m = sums1 + s2_m
    total_p = sums1 + s2_p
    
    abs_total_m = np.abs(total_m)
    abs_total_p = np.abs(total_p)
    
    mask_m = total_m != 0
    mask_p = total_p != 0
    
    abs_total_m[~mask_m] = np.iinfo(np.int64).max
    abs_total_p[~mask_p] = np.iinfo(np.int64).max
    
    min_abs_m = np.min(abs_total_m)
    min_abs_p = np.min(abs_total_p)
    
    # Combine all 4 candidates (l, r, m, p) - note r is idx, l is idx-1, m is idx-1, p is idx+1
    # So we have idx-1 (l/m), idx (r), idx+1 (p).
    # We already computed min for l and r.
    # Let's just take the min of the 3 unique indices: idx-1, idx, idx+1.
    
    # Re-evaluate min from the 3 arrays
    # We have abs_total_l (idx-1), abs_total_r (idx), abs_total_p (idx+1)
    
    # Find the global min among these three
    # We can just concatenate? No, memory.
    # Just compare mins.
    
    candidates = [
        (min_abs_l, 'l', abs_total_l, idx_best_l if 'idx_best_l' in locals() else None), # Need to track idx
        (min_abs_r, 'r', abs_total_r, idx_best_r if 'idx_best_r' in locals() else None),
        (min_abs_p, 'p', abs_total_p, None)
    ]
    
    # We lost the indices for l and r in the previous block because we only stored the min value.
    # Let's re-calculate the best index for the overall minimum.
    
    # To be safe and simple, let's just find the argmin of the combined "min per index" logic?
    # No, let's just find the best s1 index from the three arrays.
    
    # We can create a 3 x N array of absolute values? 3 * 14.3M * 8 bytes = 343 MB. Might be tight but okay.
    # Or just do it in 3 passes.
    
    # Pass 1: idx-1
    if min_abs_l < best_abs_S:
        best_abs_S = min_abs_l
        idx_best_s1 = np.argmin(abs_total_l)
        s1_val = sums1[idx_best_s1]
        s2_val = s2_l[idx_best_s1]
        
    # Pass 2: idx
    if min_abs_r < best_abs_S:
        best_abs_S = min_abs_r
        idx_best_s1 = np.argmin(abs_total_r)
        s1_val = sums1[idx_best_s1]
        s2_val = s2_r[idx_best_s1]
        
    # Pass 3: idx+1
    if min_abs_p < best_abs_S:
        best_abs_S = min_abs_p
        idx_best_s1 = np.argmin(abs_total_p)
        s1_val = sums1[idx_best_s1]
        s2_val = s2_p[idx_best_s1]
        
    if best_abs_S == np.iinfo(np.int64).max:
        # Fallback: all sums are zero? Impossible.
        # Or maybe we missed it.
        # Return a simple valid solution
        return [1, -1] + [0] * 28
        
    # Now we have s1_val and s2_val that minimize |s1+s2| != 0.
    # We need to reconstruct the coefficients.
    
    def reconstruct_coeffs(weights_half, target_sum):
        """
        Find coefficients in {-1, 0, 1} for weights_half that sum to target_sum.
        Uses backtracking/DFS.
        """
        n_half = len(weights_half)
        coeffs = [0] * n_half
        current_sum = 0
        
        # To make DFS efficient, we can use the fact that we generated the sums iteratively.
        # But we didn't store the paths.
        # We can just do a standard DFS with pruning?
        # 3^15 is 14M. DFS might be slow if not careful.
        # However, we only need ONE solution.
        # We can use a set of reachable sums for the remaining weights?
        
        # Better: Meet-in-the-middle for reconstruction?
        # Or just simple DFS with memoization of (index, remaining_sum)?
        # remaining_sum can be large.
        
        # Let's try a simple DFS.
        # Since n_half is 15, 3^15 is 14M.
        # A recursive DFS in Python is slow.
        # Iterative DFS?
        
        # Alternative: Since we have the list of all sums for this half (we generated it),
        # we can store the "path" information during generation?
        # That would require 14M * 15 integers, which is too much memory (1.7GB).
        
        # Let's use a randomized approach?
        # Or just simple DFS with a limit?
        
        # Actually, we can use the property that the weights are distinct and large.
        # Let's try a simple recursive DFS.
        
        def dfs(idx, curr_sum):
            if idx == n_half:
                return curr_sum == target_sum
            for c in [-1, 0, 1]:
                new_sum = curr_sum + c * weights_half[idx]
                if dfs(idx + 1, new_sum):
                    coeffs[idx] = c
                    return True
            return False
            
        # This is too slow for 15 depth in Python if it has to explore many branches.
        # But we know a solution exists.
        # Let's try to optimize by ordering?
        
        # A better way:
        # We can store the "choice" for each sum in a dictionary?
        # sum -> (prev_sum, choice)
        # This is BFS/DFS storing predecessors.
        # Memory: 14M entries. Each entry (int, int).
        # Python dict overhead is high.
        
        # Let's stick to the vectorized generation but add path tracking?
        # No, too much memory.
        
        # Let's use a different reconstruction method:
        # We can use the fact that we can generate the sums again, but this time storing the coefficients?
        # No.
        
        # Let's just use a simple greedy or randomized search for reconstruction?
        # No, must be exact.
        
        # Let's try to use the "predecessor" map but with arrays?
        # We can store for each sum in the final array, which index in the previous array it came from, and what choice was made.
        # This requires 14M * 15 * (index + choice) ~ 14M * 15 * 8 bytes ~ 1.7GB. Too much.
        
        # How about we only store the path for the specific target?
        # We can do a backward search from the target?
        
        # Let's try a simple recursive DFS with pruning based on min/max possible sum of remaining weights.
        
        # Precompute suffix min/max sums?
        # Max sum from i to end is sum(weights[i:])
        # Min sum from i to end is -sum(weights[i:])
        
        suffix_sums = np.zeros(n_half + 1, dtype=np.int64)
        for i in range(n_half - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + weights_half[i]
            
        def dfs_opt(idx, curr_sum, target):
            if idx == n_half:
                return curr_sum == target
            # Pruning:
            # curr_sum + c*w[idx] + remaining_sum must be able to reach target
            # remaining_sum is in [-suffix_sums[idx+1], suffix_sums[idx+1]]
            for c in [-1, 0, 1]:
                new_sum = curr_sum + c * weights_half[idx]
                # Check if target is reachable from new_sum with remaining weights
                diff = target - new_sum
                if -suffix_sums[idx+1] <= diff <= suffix_sums[idx+1]:
                    if dfs_opt(idx + 1, new_sum, target):
                        coeffs[idx] = c
                        return True
            return False
            
        if dfs_opt(0, 0, target_sum):
            return coeffs
        else:
            return None # Should not happen
            
    coeffs1 = reconstruct_coeffs(w1, s1_val)
    coeffs2 = reconstruct_coeffs(w2, s2_val)
    
    if coeffs1 is None or coeffs2 is None:
        # Fallback
        return [1, -1] + [0] * 28
        
    return coeffs1 + coeffs2

# EVOLVE-BLOCK-END
