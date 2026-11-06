"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    small_prime_set = set(small_primes)
    
    # We need to compute the "23-smooth part" for numbers in [1001, 1000024]
    # The 23-smooth part of m is the largest divisor of m composed only of primes < 24
    
    # Strategy: For each number m, repeatedly divide by small primes to extract the smooth part
    # Then for each n, count distinct smooth parts among n+1..n+24
    
    # Precompute smooth parts for a range
    # Range: n in [1000, 1000000], so n+i in [1001, 1000024]
    
    max_val = 1000024
    
    # Compute smooth part for all numbers up to max_val
    # smooth_part[m] = m with all prime factors >= 24 removed
    smooth_part = np.zeros(max_val + 1, dtype=np.int64)
    
    for m in range(1, max_val + 1):
        x = m
        for p in small_primes:
            while x % p == 0:
                x //= p
        # x is now the part with primes >= 24
        # smooth part = m // x
        smooth_part[m] = m // x
    
    # Now search for n minimizing distinct smooth parts in window [n+1, n+24]
    best_n = 1000
    best_count = 999999
    
    # Use a sliding window approach
    # For efficiency, we can use a frequency count
    
    # Let's do a direct search but with early termination
    # First pass: check all n from 1000 to 1000000
    # For each n, count distinct smooth parts in smooth_part[n+1:n+25]
    
    # To speed up, let's use a set-based approach
    # But that's O(24) per n, so 24 * 1e6 = 24e6 operations, should be fine
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Get smooth parts for n+1 to n+24
        window = smooth_part[n+1:n+25]
        # Count distinct values
        distinct_count = len(set(window))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
            # If we found a very low count, we can try to continue to see if we can do better
            # Minimum possible is 1 (all have same smooth part)
            if best_count == 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
