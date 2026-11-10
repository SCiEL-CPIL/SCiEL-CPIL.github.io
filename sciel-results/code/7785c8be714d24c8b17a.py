"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using sieve-like approach
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    # Start with 1 for all, then for each small prime p, multiply p into all multiples
    small_factor = [1] * (max_val + 1)
    small_factor[0] = 0
    
    for p in small_primes:
        # For each multiple of p, we need to extract ALL powers of p
        # and multiply them into small_factor
        # Process by dividing out p repeatedly
        for multiple in range(p, max_val + 1, p):
            val = multiple
            while val % p == 0:
                small_factor[multiple] *= p
                val //= p
    
    # Now slide window of 24 over n in [1000, 1000000]
    # For candidate n, the numbers are n+1, n+2, ..., n+24
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Iterate over valid n
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        # Extract factors for n+1 .. n+24
        # Use a set to count distinct
        factors = set()
        for i in range(1, 25):
            factors.add(small_factor[n + i])
        count = len(factors)
        if count < best_score:
            best_score = count
            best_n = n
            if best_score == 1:
                break  # can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
