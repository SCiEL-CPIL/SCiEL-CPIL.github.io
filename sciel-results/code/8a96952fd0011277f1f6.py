import numpy as np
import time
from collections import defaultdict
import bisect

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    N = P['maximum']
    target_ratio = 1.5
    
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 5s margin
    
    # 1. Compute sigma(n) for n in [1, N]
    # Use SPF and multiplicative DP for speed and memory efficiency
    
    # SPF sieve
    spf = np.arange(N + 1, dtype=np.int64)
    # Mark composites
    for i in range(2, int(np.sqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            # Mark multiples
            start = i * i
            step = i
            spf[start::step] = np.where(spf[start::step] == 0, i, spf[start::step])
            # Wait, spf init is arange, so spf[i] == i for primes.
            # For composites, spf[j] should be the smallest prime factor.
            # The standard sieve:
            # spf = np.zeros(N+1)
            # for i in range(2, N+1):
            #   if spf[i] == 0:
            #     spf[i::i] = i
            # This is the slow loop.
            
            # Let's use the standard efficient SPF sieve in NumPy:
            # We can't easily do the "if spf[i]==0" check in a vectorized way for the outer loop without a Python loop.
            # But the outer loop only goes to sqrt(N) ~ 3162.
            # So we only have 3162 iterations!
            # This is MUCH faster than 10 million iterations.
            
    # Re-implement SPF sieve correctly
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
            
    # This loop is 10 million iterations, but the inner `spf[i::i] = i` is only executed for primes (~660k times).
    # The check `if spf[i] == 0` is cheap.
    # This is still a Python loop of 10M.
    # Can we do better?
    # Yes, only iterate up to N, but the assignment is sparse.
    # Actually, the previous additive sieve had 10M iterations with heavy work.
    # This SPF sieve has 10M iterations with light work (check) and sparse heavy work (assignment for primes).
    # It should be faster.
    
    # DP for Sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # We need:
    # rest[i]: n / (highest power of spf[n] dividing n)
    # p_pow_sum[i]: sum of divisors of the highest power of spf[n] dividing n
    # last_term[i]: highest power of spf[n] dividing n
    
    rest = np.zeros(N + 1, dtype=np.int64)
    p_pow_sum = np.zeros(N + 1, dtype=np.int64)
    last_term = np.zeros(N + 1, dtype=np.int64)
    
    rest[1] = 1
    p_pow_sum[1] = 1
    last_term[1] = 1
    
    for i in range(2, N + 1):
        p = spf[i]
        if i % (p * p) != 0:
            rest[i] = i // p
            last_term[i] = p
            p_pow_sum[i] = 1 + p
        else:
            m = i // p
            rest[i] = rest[m]
            last_term[i] = last_term[m] * p
            p_pow_sum[i] = p_pow_sum[m] + last_term[i]
        
        sigma[i] = sigma[rest[i]] * p_pow_sum[i]
        
    if time.time() - start_time > timeout:
        return [11, 6]

    # 2. Group indices by sigma value
    # Sort indices by sigma value
    # indices = np.arange(1, N + 1)
    # order = np.argsort(sigma[1:], kind='stable') + 1
    # This creates a copy of sigma[1:] and sorts.
    
    # To save memory, we can sort the indices directly using the sigma values.
    indices = np.arange(1, N + 1, dtype=np.int64)
    # np.argsort with key? No, key is not supported for argsort in older numpy?
    # Actually, `np.argsort(sigma[indices])` works.
    
    sorted_indices = indices[np.argsort(sigma[indices], kind='stable')]
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() - start_time > timeout:
        return [11, 6]

    # 3. Find best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate through sorted array to find blocks of equal sigma
    # Block starts at i, ends at j where sigma changes.
    # We can find boundaries using np.diff
    
    # Create a mask of where sigma changes
    # changes = np.diff(sorted_sigmas) != 0
    # boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [len(sorted_sigmas)]))
    
    # Instead of creating these arrays, we can just iterate?
    # No, iterating 10M times in Python is slow.
    # Let's use the boundaries approach.
    
    changes = np.diff(sorted_sigmas) != 0
    # boundaries are indices in sorted_indices where the value changes
    # start of block: 0, and indices where changes is True + 1
    # end of block: indices where changes is True, and len
    
    # We need to process each block.
    # Number of blocks is ~ N / log N? Or less.
    # If we have 100k blocks, iterating 100k times is fine.
    
    # Get start and end indices for each block
    # starts = np.concatenate(([0], np.where(changes)[0] + 1))
    # ends = np.concatenate((np.where(changes)[0] + 1, [len(sorted_sigmas)]))
    
    # This creates arrays of size ~100k. Fast.
    
    starts = np.concatenate(([0], np.where(changes)[0] + 1))
    ends = np.concatenate((np.where(changes)[0] + 1, [len(sorted_sigmas)]))
    
    # Iterate over blocks
    for k in range(len(starts)):
        s_start = starts[k]
        s_end = ends[k]
        
        if s_end - s_start < 2:
            continue
            
        if time.time() - start_time > timeout:
            break
            
        # Get the numbers in this block
        # They are sorted because sorted_indices is sorted by sigma, and within same sigma,
        # np.argsort with kind='stable' preserves the original order of indices?
        # Original indices were 1..N.
        # So yes, they are in increasing order of n.
        
        block_nums = sorted_indices[s_start:s_end]
        
        # For each b in block_nums, find a in block_nums closest to 1.5 * b
        # Since block_nums is sorted, we can use bisect.
        # But bisect works on Python lists.
        # Converting NumPy array to list is fast for small blocks.
        # Average block size?
        # If sigma values are distinct, block size is 1.
        # If not, some are large.
        
        # To avoid conversion overhead for large blocks, we can use NumPy searchsorted?
        # `np.searchsorted(block_nums, target)`
        
        for b in block_nums:
            target_a = 1.5 * b
            
            # Find insertion point
            idx = np.searchsorted(block_nums, target_a)
            
            candidates = []
            if idx < len(block_nums):
                candidates.append(block_nums[idx])
            if idx > 0:
                candidates.append(block_nums[idx - 1])
                
            for a in candidates:
                if a == b:
                    continue
                err = abs(a / b - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b

    # Ensure return types are Python int
    return [int(best_a), int(best_b)]
