"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin: 90s limit, leave 15s buffer

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size  # 1000024
    
    nvals = max_val - lower + 1  # 1000025
    
    # Precompute smooth parts using vectorized NumPy operations
    # For each number in [lower, max_val], extract the part composed of primes < 24
    
    # Start with all 1s
    smooth_parts = np.ones(nvals, dtype=np.int64)
    
    # Numbers array: lower, lower+1, ..., max_val
    nums = np.arange(lower, max_val + 1, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        # Find indices where nums is divisible by p
        # We need to repeatedly divide by p and multiply smooth_parts
        # Use a while loop for each power of p
        
        # First, get the mask of numbers divisible by p
        mask = (nums % p == 0)
        
        # We'll iteratively process powers of p
        # For p^k, the numbers divisible by p^k contribute p to smooth_parts
        # Actually, we need to count how many times p divides each number
        
        # Better approach: for each p, compute the exponent of p in each number
        # Then multiply smooth_parts by p^exponent
        
        # Let's do it iteratively: for each power p^1, p^2, p^3, ...
        # For each number divisible by p^k but not p^(k+1), the exponent is k
        
        # Simpler: repeatedly check divisibility
        temp = nums.copy()
        
        while True:
            mask = (temp % p == 0)
            if not np.any(mask):
                break
            # Multiply smooth_parts by p for all indices where temp is still divisible by p
            smooth_parts[mask] *= p
            temp[mask] //= p
        
    # Check time
    if time.time() > deadline:
        return lower
    
    # Now smooth_parts[i] = the p-smooth part of (lower + i) for p < 24
    # Map to compact IDs for faster counting
    # Use np.unique to get unique values and their IDs
    unique_vals, inverse = np.unique(smooth_parts, return_inverse=True)
    num_ids = len(unique_vals)
    
    sp_ids = inverse.astype(np.int32)
    
    # Sliding window to find n that minimizes distinct smooth parts in window [n+1, n+24]
    # Window at position n (0-indexed as n-lower) covers indices [n-lower+1, n-lower+24]
    # which correspond to numbers (lower + n - lower + 1) to (lower + n - lower + 24) = (n+1) to (n+24)
    
    # We need to count distinct values in sp_ids[i+1 : i+25] for i from 0 to upper-lower
    # That is, for each n from lower to upper, count distinct in sp_ids[n-lower+1 : n-lower+25]
    
    # Use a frequency array and sliding window
    freq = np.zeros(num_ids, dtype=np.int32)
    
    best_n = lower
    best_count = window_size + 1  # Max possible is 24
    current_distinct = 0
    
    # Initialize window for n=lower: indices 1 to 24 (0-indexed in sp_ids)
    for i in range(1, window_size + 1):
        val = sp_ids[i]
        if freq[val] == 0:
            current_distinct += 1
        freq[val] += 1
    
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower
    
    # Slide window
    # For n from lower+1 to upper:
    #   out_idx = n - lower (the element leaving)
    #   in_idx = n + 24 - lower (the element entering)
    
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
        
        out_idx = n - lower
        in_idx = n + 24 - lower
        
        val_out = sp_ids[out_idx]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            current_distinct -= 1
        
        val_in = sp_ids[in_idx]
        if freq[val_in] == 0:
            current_distinct += 1
        freq[val_in] += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 2:
                # 1 is impossible, so 2 is the theoretical minimum
                break
    
    return best_n
# EVOLVE-BLOCK-END
