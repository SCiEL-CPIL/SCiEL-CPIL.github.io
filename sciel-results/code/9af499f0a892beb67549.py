import numpy as np
import bisect
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # 1. Compute sigma(n) for n in [1, N]
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    for i in range(1, N + 1):
        if time.time() > deadline:
            return [11, 6]
        sigma[i::i] += i
        
    # 2. Group by sigma value using argsort
    # We want to group indices 1..N by sigma[1..N]
    # sigma[1:] is an array of length N
    try:
        # argsort with stable sort to preserve original index order? 
        # Actually, we just need the set of numbers. 
        # We will extract the numbers and sort them anyway for bisect.
        # So stability doesn't matter for the final sorted list.
        order = np.argsort(sigma[1:])
        sorted_sigma = sigma[1:][order]
        
        # Find indices where sigma value changes
        # np.diff returns array of length N-1
        # If sorted_sigma[i] == sorted_sigma[i+1], no change.
        # Change at i if sorted_sigma[i] != sorted_sigma[i+1].
        # The new group starts at i+1.
        # First group starts at 0.
        
        # To avoid creating huge intermediate arrays if possible, 
        # but N=10^7 is manageable for memory (80MB per int64 array).
        
        # diff = np.diff(sorted_sigma)
        # change_indices = np.where(diff != 0)[0]
        # group_starts = np.concatenate(([0], change_indices + 1))
        # group_ends = np.concatenate((change_indices + 1, [N]))
        
        # This creates arrays of size ~ #groups.
        
        # Alternative: Loop through sorted_sigma and find boundaries?
        # Python loop over N is slow.
        # Let's use the np.where approach.
        
        diff = np.diff(sorted_sigma)
        change_indices = np.where(diff != 0)[0]
        
        # Add sentinels
        starts = np.concatenate(([0], change_indices + 1))
        ends = np.concatenate((change_indices + 1, [N]))
        
    except Exception:
        return [11, 6]

    best_err = float('inf')
    best_a = 11
    best_b = 6
    
    # 3. Search for best pair
    for k in range(len(starts)):
        if time.time() > deadline:
            break
            
        s_start = starts[k]
        s_end = ends[k]
        
        if s_end - s_start < 2:
            continue
            
        # Extract the numbers in this group
        # order[s_start:s_end] gives the 0-based indices in the original array (1..N)
        # So the numbers are order[s_start:s_end] + 1
        nums = order[s_start:s_end] + 1
        nums.sort()
        
        # For each a in nums, find b closest to 2a/3
        # We can vectorize the target calculation?
        # targets = (2 * nums) / 3.0
        # But we need to find nearest in nums for each target.
        # np.searchsorted(nums, targets) gives the indices.
        
        targets = (2 * nums) / 3.0
        # searchsorted returns index where target would be inserted to maintain order
        # i.e., nums[idx-1] <= target < nums[idx]
        # So candidates are idx-1 and idx.
        
        idxs = np.searchsorted(nums, targets)
        
        # We need to check idxs and idxs-1.
        # Handle boundaries: idxs can be 0 or len(nums).
        
        # Candidate 1: nums[idxs] (if idxs < len)
        # Candidate 2: nums[idxs-1] (if idxs > 0)
        
        # To avoid out of bounds, we can clip indices or use masks.
        # Let's create arrays of candidate indices.
        
        # Mask for valid right candidates (idxs < len)
        mask_right = idxs < len(nums)
        # Mask for valid left candidates (idxs > 0)
        mask_left = idxs > 0
        
        # Initialize best error for this group to inf
        group_best_err = float('inf')
        group_best_a = -1
        group_best_b = -1
        
        # Process right candidates
        if np.any(mask_right):
            valid_idxs = idxs[mask_right]
            # The 'a' values corresponding to these targets
            # Note: targets was computed from nums (all of them).
            # So the 'a' values are nums[mask_right]
            a_vals = nums[mask_right]
            b_vals = nums[valid_idxs]
            
            # Calculate error |a/b - 1.5|
            # Vectorized
            errs = np.abs(a_vals / b_vals - 1.5)
            min_idx = np.argmin(errs)
            if errs[min_idx] < group_best_err:
                group_best_err = errs[min_idx]
                group_best_a = a_vals[min_idx]
                group_best_b = b_vals[min_idx]
                
        # Process left candidates
        if np.any(mask_left):
            valid_idxs = idxs[mask_left] - 1
            a_vals = nums[mask_left]
            b_vals = nums[valid_idxs]
            
            errs = np.abs(a_vals / b_vals - 1.5)
            min_idx = np.argmin(errs)
            if errs[min_idx] < group_best_err:
                group_best_err = errs[min_idx]
                group_best_a = a_vals[min_idx]
                group_best_b = b_vals[min_idx]
                
        if group_best_a != -1 and group_best_err < best_err:
            best_err = group_best_err
            best_a = group_best_a
            best_b = group_best_b
            
    return [int(best_a), int(best_b)]
