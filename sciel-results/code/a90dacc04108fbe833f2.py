import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = 50.0
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n <= N
    sigma = np.zeros(N + 1, dtype=np.int64)
    # This loop is O(N log N) in terms of vectorized ops, but N Python iterations.
    # It is the most time-consuming part.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() - start_time > deadline:
        return [11, 6]
        
    # 2. Sort indices by sigma value
    # We want to group numbers with the same sigma
    sorted_indices = np.argsort(sigma)
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() - start_time > deadline:
        return [11, 6]
        
    # 3. Identify blocks of equal sigma
    # Find where sigma value changes
    diffs = np.diff(sorted_sigmas)
    change_idx = np.where(diffs != 0)[0]
    
    starts = np.concatenate(([0], change_idx + 1))
    ends = np.concatenate((change_idx + 1, [len(sorted_indices)]))
    
    sizes = ends - starts
    
    # Filter blocks with at least 2 elements
    mask = sizes >= 2
    if not np.any(mask):
        return [11, 6]
        
    starts = starts[mask]
    ends = ends[mask]
    
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)
    
    # 4. Search for best pair in each block
    for s, e in zip(starts, ends):
        # Extract numbers in this block
        nums = sorted_indices[s:e]
        
        # Sort nums to allow binary search / searchsorted
        nums = np.sort(nums)
        
        block_size = len(nums)
        
        if block_size < 100:
            # Small block: use Python loop with bisect
            nums_list = nums.tolist()
            for a in nums_list:
                target = a / 1.5
                pos = bisect_left(nums_list, target)
                
                # Check pos
                if pos < block_size:
                    b = nums_list[pos]
                    if a != b:
                        err = abs(a / b - 1.5)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            
                # Check pos - 1
                if pos > 0:
                    b = nums_list[pos - 1]
                    if a != b:
                        err = abs(a / b - 1.5)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
        else:
            # Large block: vectorized
            targets = nums / 1.5
            pos = np.searchsorted(nums, targets)
            
            # Clip pos to valid range [1, block_size - 1] to access pos-1 and pos safely
            # If pos is 0, we only check pos (which is 0). But we want pos-1 too?
            # If pos is 0, target < nums[0]. Closest is nums[0].
            # If pos is block_size, target > nums[-1]. Closest is nums[-1].
            # np.searchsorted returns insertion point.
            # If we clip to [1, block_size-1], we might miss the ends.
            # Better to handle ends or just check pos and pos-1 with bounds.
            
            # Let's just check pos and pos-1 with bounds in vectorized way?
            # It's easier to just compute errors for pos and pos-1 where valid.
            
            # Create arrays for b1 and b2
            # b1 = nums[pos]
            # b2 = nums[pos-1]
            
            # We need to be careful with out of bounds.
            # pos can be 0 .. block_size
            # If pos == 0, b1 = nums[0], b2 invalid.
            # If pos == block_size, b1 invalid, b2 = nums[-1].
            
            # Let's use np.take with mode='clip' or just create safe indices.
            pos_safe = np.clip(pos, 0, block_size - 1)
            pos_prev = np.clip(pos - 1, 0, block_size - 1)
            
            b1 = nums[pos_safe]
            b2 = nums[pos_prev]
            
            err1 = np.abs(nums / b1 - 1.5)
            err2 = np.abs(nums / b2 - 1.5)
            
            # If pos was 0, b2 is nums[0] (same as b1), so err2 == err1.
            # If pos was block_size, b1 is nums[-1] (same as b2), so err1 == err2.
            # So taking min is safe.
            
            min_errs = np.minimum(err1, err2)
            min_idx = np.argmin(min_errs)
            
            if min_errs[min_idx] < best_err:
                a = int(nums[min_idx])
                # Determine which b was better
                if err1[min_idx] < err2[min_idx]:
                    b = int(b1[min_idx])
                else:
                    b = int(b2[min_idx])
                
                # Ensure a != b if we want to avoid trivial 1.0 ratio?
                # The problem doesn't forbid a=b, but 11,6 is better than 1,1 (err 0.5 vs 0.33).
                # If a=b, err=0.5.
                if a == b:
                    # If best_err is already < 0.5, this won't update.
                    # If best_err was 0.5 (initial 11,6 is 0.33), so a=b won't update.
                    pass
                
                best_err = min_errs[min_idx]
                best_a = a
                best_b = b
                
        if time.time() - start_time > deadline:
            break
            
    return [int(best_a), int(best_b)]
