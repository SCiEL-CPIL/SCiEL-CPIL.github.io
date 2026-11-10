"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute the extracted factor (23-smooth part) for each number
    # The extracted factor is the product of p^k for all primes p < 24 where p^k || x
    min_start = 1001
    max_end = 1000000 + 24
    
    # Precompute extracted factor for each number using sieve-like approach
    # For each prime p, for each multiple of p, multiply the factor by the highest power of p
    num_range = max_end - min_start + 1
    extracted = [1] * (num_range + 1)
    
    for p in primes:
        # For each multiple of p in range, find the highest power of p dividing it
        # and multiply extracted by that power
        # Start from the first multiple of p >= min_start
        first_mult = ((min_start + p - 1) // p) * p
        for x in range(first_mult, max_end + 1, p):
            # Find highest power of p dividing x
            temp = x
            pk = 1
            while temp % p == 0:
                pk *= p
                temp //= p
            extracted[x - min_start] *= pk
    
    # Now extracted[i] holds the 23-smooth part of (min_start + i)
    
    # Sliding window over extracted values
    # We need to count distinct values in a window of 24 consecutive extracted values
    # Extracted values can be up to ~1000024, so use a dict for frequency
    
    best_n = 1000
    best_count = float('inf')
    
    # Use a dict for frequency counting of extracted values
    freq = {}
    
    # Initialize window for n=1000: numbers 1001 to 1024
    # indices 0 to 23 in extracted array
    for i in range(24):
        val = extracted[i]
        freq[val] = freq.get(val, 0) + 1
    
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    # For n from 1001 to 999976:
    # Remove extracted[n - min_start] (which is extracted[n-1001])
    # Add extracted[n + 24 - min_start] (which is extracted[n+24-1001])
    for n in range(1001, 999977):
        # Check deadline
        if time.time() > deadline - 2:
            break
        
        idx_remove = n - min_start  # = n - 1001
        val_remove = extracted[idx_remove]
        freq[val_remove] -= 1
        if freq[val_remove] == 0:
            del freq[val_remove]
            distinct -= 1
        
        idx_add = n + 24 - min_start  # = n + 24 - 1001
        val_add = extracted[idx_add]
        if val_add not in freq:
            distinct += 1
        freq[val_add] = freq.get(val_add, 0) + 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
