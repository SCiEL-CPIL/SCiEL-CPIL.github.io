"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to compute the "23-smooth part" for numbers in [1001, 1000024]
    # The 23-smooth part of m is the largest divisor of m composed only of primes < 24
    
    max_val = 1000024
    
    # Compute smooth part for all numbers up to max_val using a sieve approach
    # smooth_part[m] = m with all prime factors >= 24 removed
    # Equivalently: smooth_part[m] = m / rough_part[m] where rough_part has only primes >= 24
    
    # Efficient approach: for each small prime p, divide p out of all multiples
    # Start with smooth_part[m] = m, then for each small prime p, 
    # we DON'T divide - instead we want to KEEP the small prime factors
    # So: rough_part[m] = m, then for each small prime p, divide p out of rough_part[multiples]
    
    rough_part = np.arange(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For all multiples of p, divide out all factors of p
        for multiple in range(p, max_val + 1, p):
            while rough_part[multiple] % p == 0:
                rough_part[multiple] //= p
    
    # Now smooth_part[m] = m // rough_part[m]
    smooth_part = np.zeros(max_val + 1, dtype=np.int64)
    for m in range(1, max_val + 1):
        smooth_part[m] = m // rough_part[m]
    
    # Now search for n minimizing distinct smooth parts in window [n+1, n+24]
    best_n = 1000
    best_count = 999999
    
    # Use a sliding window approach with frequency counting
    # For efficiency, we track counts of each smooth part value in the current window
    
    # Since smooth parts can be large, we use a dictionary for frequency
    # But for a window of 24, we can just use a set
    
    # First, let's do a quick pass to find a good candidate
    # Then refine if time permits
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Get smooth parts for n+1 to n+24
        window = smooth_part[n+1:n+25]
        # Count distinct values
        distinct_count = len(set(window.tolist()))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
            # If we found a very low count, we can try to continue
            # Minimum possible is 1 (all have same smooth part)
            if best_count == 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
