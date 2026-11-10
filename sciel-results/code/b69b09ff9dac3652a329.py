"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using efficient sieve-like approach
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    small_factor[0] = 0
    
    for p in small_primes:
        # For each multiple of p, multiply p into small_factor for each power
        multiples = np.arange(p, max_val + 1, p)
        for m in multiples:
            val = m
            while val % p == 0:
                small_factor[m] *= p
                val //= p
    
    # Convert to list for faster access in sliding window
    sf = small_factor.tolist()
    
    # Sliding window approach with frequency counting
    # For candidate n, numbers are n+1 .. n+24
    
    best_score = 25
    best_n = 1000
    
    # Use a dictionary for frequency counting
    freq = {}
    distinct = 0
    
    # Initialize window for n=1000: factors of 1001..1024
    for i in range(1, 25):
        f = sf[1000 + i]
        if f not in freq:
            freq[f] = 1
            distinct += 1
        else:
            freq[f] += 1
    
    if distinct < best_score:
        best_score = distinct
        best_n = 1000
    
    # Slide window: for n from 1001 to 1000000
    # Remove n, add n+24
    # Check time every 5000 iterations to reduce overhead
    check_interval = 5000
    last_check = start_time
    
    for n in range(1001, 1000001):
        # Remove element at position n
        remove_f = sf[n]
        freq[remove_f] -= 1
        if freq[remove_f] == 0:
            del freq[remove_f]
            distinct -= 1
        
        # Add element at position n+24
        add_f = sf[n + 24]
        if add_f not in freq:
            freq[add_f] = 1
            distinct += 1
        else:
            freq[add_f] += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 1:
                break
        
        # Periodic time check
        if n % check_interval == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
