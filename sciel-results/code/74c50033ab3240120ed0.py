import math
import time
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 120.0  # seconds
    MAX_S = 10**12
    TIME_MARGIN = 10.0 # Reserve time for final check and safety

    # 1. Generate primes up to 10^6 for trial division / sieve limits
    # We need primes up to sqrt(MAX_S) = 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Segmented Sieve for Phi using NumPy for vectorization
    # Computes phi for all n in [start, end]
    def segmented_phi(start, end):
        size = end - start + 1
        if size <= 0:
            return []
        
        # Initialize nums and phis as numpy arrays
        # nums[i] holds the current value of the number (start+i) after removing prime factors
        # phis[i] holds the current value of phi(start+i)
        nums = np.arange(start, end + 1, dtype=np.int64)
        phis = nums.copy()
        
        # Sieve with primes up to sqrt(end)
        for p in primes:
            if p * p > end:
                break
                
            # Find the first multiple of p in [start, end]
            first_mult = ((start + p - 1) // p) * p
            
            if first_mult > end:
                continue
                
            # Create indices of multiples of p in [start, end]
            # start_idx is 0-based index in the block
            start_idx = first_mult - start
            
            # Create an array of indices
            idxs = np.arange(start_idx, size, p)
            
            # Check which of these are actually divisible by p in the *current* nums state
            mask = nums[idxs] % p == 0
            
            if not np.any(mask):
                continue
            
            # Get the indices that need updating
            update_idxs = idxs[mask]
            
            # Update phi: phis[idx] -= phis[idx] // p
            phis[update_idxs] -= phis[update_idxs] // p
            
            # Remove all factors of p from nums[update_idxs]
            current_nums = nums[update_idxs].copy()
            while True:
                div_mask = current_nums % p == 0
                if not np.any(div_mask):
                    break
                current_nums[div_mask] //= p
            nums[update_idxs] = current_nums
            
        # Handle remaining prime factors > sqrt(end)
        remaining = nums > 1
        if np.any(remaining):
            rem_idxs = np.where(remaining)[0]
            phis[rem_idxs] = phis[rem_idxs] - phis[rem_idxs] // nums[rem_idxs]
                
        return phis.tolist()

    # 3. Helper to find longest run of distinct values
    def find_longest_distinct_run(phis, block_start):
        seen_vals = {}  # value -> index
        l = 0
        max_len = 0
        max_start_idx = 0
        
        for r in range(len(phis)):
            v = phis[r]
            if v in seen_vals:
                prev_idx = seen_vals[v]
                if prev_idx >= l:
                    l = prev_idx + 1
            
            seen_vals[v] = r
            
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start_idx = l
            
            # Early exit if we hit the cap
            if max_len >= 200:
                return 200, block_start + max_start_idx
        
        return max_len, block_start + max_start_idx

    # 4. Search Strategy
    # We know that long runs are more likely to be found in the higher ranges.
    # We will scan downwards from 10^12 in large blocks.
    
    best_k = 1
    best_s = 1
    
    # Block size: 200,000. 
    # Larger blocks allow finding longer runs if they span the previous block boundary.
    # Time per block: ~1-2s. 
    # In 120s, we can scan ~60-100 blocks, covering 10^7 to 2*10^7 range.
    
    block_size = 200000
    current_end = MAX_S
    
    # We can also try some specific "hotspots" if time permits, 
    # but a dense scan of the top range is a good heuristic.
    
    while current_end > 1:
        elapsed = time.time() - start_time
        if elapsed > LIMIT - TIME_MARGIN:
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        phis = segmented_phi(block_start, current_end)
        k, s = find_longest_distinct_run(phis, block_start)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k >= 200:
            return {'start': best_s, 'length': best_k}
        
        # Move to previous block
        current_end = block_start - 1
        
    # If we didn't find a very long run in the top 10^7, 
    # maybe we should try some random samples in the middle?
    # Or just return the best found.
    # Given the time, the top scan is the most robust.
    
    return {'start': best_s, 'length': best_k}
