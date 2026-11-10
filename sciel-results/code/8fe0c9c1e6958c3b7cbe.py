"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    nprimes = len(primes)  # 9
    max_val = 1000000 + 24  # n can be up to 1000000, so n+24 up to 1000024
    
    # Precompute bitmask for each number: which small primes divide it
    # bitmask[i] = bitmask of primes that divide i
    bitmask = np.zeros(max_val + 1, dtype=np.int32)
    
    for pi, p in enumerate(primes):
        mask_bit = 1 << pi
        for multiple in range(p, max_val + 1, p):
            bitmask[multiple] |= mask_bit
    
    # Now slide a window of 24 consecutive numbers (n+1 to n+24)
    # and count distinct bitmasks
    # For each position i (1-indexed in the window), we look at bitmask[i]
    # We want to minimize the number of distinct values in bitmask[n+1..n+24]
    
    best_n = 1000
    best_count = 25  # maximum possible is 24 distinct
    
    # Use a frequency counter for the sliding window
    # Since bitmask values are at most 2^9 - 1 = 511, we can use an array of size 512
    freq = np.zeros(512, dtype=np.int32)
    
    # Initialize window for n=1000: numbers 1001 to 1024
    start = 1001
    end = 1001 + 24  # exclusive
    for i in range(start, end):
        freq[bitmask[i]] += 1
    distinct = np.count_nonzero(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000000):
        # Remove n, add n+24
        old_val = bitmask[n]
        new_val = bitmask[n + 24]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct -= 1
        freq[new_val] += 1
        if freq[new_val] == 1:
            distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
