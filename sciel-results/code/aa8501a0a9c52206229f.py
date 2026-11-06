"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    """
    Find n in [1000, 1000000] such that for each n+i (1<=i<=24),
    we extract the smooth part (product of prime powers for primes < 24),
    and minimize the number of DISTINCT smooth parts across all 24 values.
    
    Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    
    Strategy: Precompute the smooth part for all numbers up to 1000000+24,
    then slide a window of 24 and count distinct smooth parts using an
    efficient sliding window with a frequency dictionary.
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute smooth part for each number
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            power = 1
            temp = multiple
            while temp % p == 0:
                power *= p
                temp //= p
            smooth[multiple] *= power
    
    # Now slide window: for n in [1000, 1000000], look at smooth[n+1]..smooth[n+24]
    # Count distinct values in that window using sliding window with frequency dict
    best_n = 1000
    best_count = float('inf')
    
    n_start = 1000
    n_end = 1000000
    
    # Initialize frequency dict for first window [n_start+1, n_start+24]
    freq = {}
    for i in range(1, 25):
        val = int(smooth[n_start + i])
        freq[val] = freq.get(val, 0) + 1
    
    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = n_start
    
    # Slide the window
    for n in range(n_start + 1, n_end + 1):
        # Remove element leaving: smooth[n]
        val_out = int(smooth[n])
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
        
        # Add element entering: smooth[n+24]
        val_in = int(smooth[n + 24])
        freq[val_in] = freq.get(val_in, 0) + 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
