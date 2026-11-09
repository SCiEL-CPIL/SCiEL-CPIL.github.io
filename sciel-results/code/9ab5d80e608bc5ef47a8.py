import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    N = 10000000
    
    # 1. Compute sigma(n) for n=1..N
    # Using numpy vectorized addition
    sigma = np.zeros(N+1, dtype=np.int64)
    for i in range(1, N+1):
        if time.time() > deadline:
            break
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]

    # 2. Sort indices by their sigma value
    indices = np.arange(1, N+1, dtype=np.int64)
    # argsort on sigma[1:]
    # We need to sort indices 1..N based on sigma[1..N]
    sorted_indices = indices[np.argsort(sigma[1:], kind='mergesort')]
    sorted_sigmas = sigma[sorted_indices]
    
    # 3. Find group boundaries
    # diff of 0 means same sigma
    # diff != 0 means new group starts after this index
    diff = np.diff(sorted_sigmas)
    boundaries = np.where(diff != 0)[0]
    
    # Create start and end indices for each group
    # Group k is from start[k] to end[k]
    # starts are 0, boundaries+1
    # ends are boundaries, N
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [N]))
    sizes = ends - starts
    
    # Filter to groups with size > 1
    valid_mask = sizes > 1
    valid_starts = starts[valid_mask]
    valid_ends = ends[valid_mask]
    
    best_err = float('inf')
    best_a, best_b = 11, 6
    
    # 4. Iterate over valid groups
    for idx in range(len(valid_starts)):
        if time.time() > deadline:
            break
            
        s = valid_starts[idx]
        e = valid_ends[idx]
        
        # Extract numbers in this group
        # sorted_indices is the array of n values
        nums = sorted_indices[s:e]
        
        # nums needs to be sorted by value for bisect/searchsorted
        # Is it already sorted? No, argsort was on sigma.
        # However, mergesort is stable, so if sigma is same, original order is kept?
        # Original order of indices is 1, 2, 3...
        # So yes, within same sigma, indices are increasing.
        # So nums is already sorted by value.
        
        # Search for best pair
        # For each b in nums, find a closest to 1.5*b
        # Since nums is sorted, we can use searchsorted
        
        # Optimization: if group is very large, this loop might be slow.
        # But typically groups are small.
        
        for b in nums:
            target = 1.5 * b
            # Find position
            pos = np.searchsorted(nums, target)
            
            # Check candidates
            # pos, pos-1, pos+1, pos-2
            
            # We must ensure we don't access out of bounds
            cands = []
            if pos < len(nums):
                cands.append(nums[pos])
            if pos > 0:
                cands.append(nums[pos-1])
            if pos + 1 < len(nums):
                cands.append(nums[pos+1])
            if pos - 2 >= 0:
                cands.append(nums[pos-2])
                
            for a in cands:
                if a == b:
                    continue
                # Calculate error
                # We want min(|a/b - 1.5|, |b/a - 1.5|)
                # a, b are numpy ints
                a_f = float(a)
                b_f = float(b)
                
                err1 = abs(a_f / b_f - 1.5)
                err2 = abs(b_f / a_f - 1.5)
                err = min(err1, err2)
                
                if err < best_err:
                    best_err = err
                    if err1 < err2:
                        best_a = int(a)
                        best_b = int(b)
                    else:
                        best_a = int(b)
                        best_b = int(a)
                        
    return [best_a, best_b]
