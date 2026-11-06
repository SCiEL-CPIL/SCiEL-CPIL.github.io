"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    N = max_val - lower + 1

    # We need to compute the small-prime part for numbers in [lower, max_val]
    # Let's create an array of these numbers
    nums = np.arange(lower, max_val + 1, dtype=np.int64)
    
    # Initialize the result array to 1
    smooth_parts = np.ones(N, dtype=np.int64)
    
    # Small primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # For each small prime, extract its contribution to the smooth part
    for p in small_primes:
        # Create a copy of the current nums to track divisions
        # Actually, we can work directly on a temporary array
        temp = nums.copy()
        factor = np.ones(N, dtype=np.int64)
        
        # While any element in temp is divisible by p
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            factor[mask] *= p
            
        smooth_parts *= factor

    # Now smooth_parts[i] is the small-prime part of (lower + i)
    # We need to slide a window of size 24 over smooth_parts
    # For n in [lower, upper], the window covers indices:
    # start_idx = (n + 1) - lower
    # end_idx = (n + 24) - lower
    # Note: smooth_parts is 0-indexed, so element at index k corresponds to number (lower + k)
    
    best_n = lower
    best_count = 25  # Max possible distinct values in a window of 24 is 24
    
    # Frequency dictionary for the current window
    freq = {}
    distinct = 0
    
    # Initialize window for n = lower
    # Window covers numbers lower+1 to lower+24
    # Indices in smooth_parts: 1 to 24
    for i in range(1, 25):
        v = int(smooth_parts[i])
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct += 1
            
    if distinct < best_count:
        best_count = distinct
        best_n = lower
        
    # Slide the window
    # n goes from lower + 1 to upper
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
            
        # Remove element at index (n - lower)
        # Previous window for n-1 covered indices [ (n-1)+1-lower, (n-1)+24-lower ] = [ n-lower, n+23-lower ]
        # Current window for n covers indices [ n+1-lower, n+24-lower ]
        # So we remove index n-lower and add index n+24-lower
        
        out_idx = n - lower
        in_idx = n + 24 - lower
        
        v_out = int(smooth_parts[out_idx])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            distinct -= 1
            
        v_in = int(smooth_parts[in_idx])
        if v_in in freq:
            freq[v_in] += 1
        else:
            freq[v_in] = 1
            distinct += 1
            
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                # Theoretical minimum, can break early
                break
                
    return best_n
# EVOLVE-BLOCK-END
