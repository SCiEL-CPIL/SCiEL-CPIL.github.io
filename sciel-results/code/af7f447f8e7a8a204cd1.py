"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_val = 1000024
    
    # Compute the "large part" (co-prime to all small primes) for each number
    # large_part[m] = m with all factors of primes < 24 removed
    # smooth_part[m] = m // large_part[m]
    
    large_part = np.arange(1, max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, divide out all factors of p
        # Process in order: for each multiple, keep dividing until not divisible
        # Use a sieve-like approach: for each power of p, mark multiples
        power = p
        while power <= max_val:
            # For each multiple of 'power', divide large_part by p
            # But we need to be careful: we want to divide out ALL factors
            # Better: for each number, while divisible by p, divide by p
            # Let's do it with a while loop per prime
            break
        
        # For each number divisible by p, repeatedly divide by p
        # We can do this in a single pass by checking divisibility
        for k in range(p, max_val + 1, p):
            while large_part[k] % p == 0:
                large_part[k] //= p
    
    smooth_part = np.arange(1, max_val + 1, dtype=np.int64) // large_part
    
    # Now search for n minimizing distinct smooth parts in window [n+1, n+24]
    best_n = 1000
    best_count = 999999
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Get smooth parts for n+1 to n+24 (1-indexed)
        window = smooth_part[n+1:n+25]
        distinct_count = len(np.unique(window))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
            if best_count == 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
