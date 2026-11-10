"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to evaluate n in [1000, 1000000]
    # For each candidate n, look at n+1 .. n+24
    # Extract the factor composed of primes strictly below 24 from each
    # Count distinct extracted factors
    
    # Precompute small-prime factor for numbers up to 1000024
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # small_factor[x] = product of prime factors < 24 (with multiplicity)
    # Use a vectorized approach with numpy for speed
    # Initialize array of 1s (the empty product)
    sf = np.ones(max_val + 1, dtype=np.int64)
    sf[0] = 0
    sf[1] = 1
    
    # For each small prime, multiply sf by p^k where k is the exponent of p in x
    # We can do this iteratively: for each prime p, for multiples of p, multiply by p
    # But we need to handle powers. Let's use a sieve-like approach.
    # Alternative: for each x, factor out small primes. But that's O(n * num_primes * log(n))
    
    # Let's use the multiplicative approach with numpy
    # For each prime p, create a mask of numbers divisible by p
    # Then divide those numbers by p and multiply sf by p
    # Repeat until no more divisions
    
    for p in small_primes:
        # Find all multiples of p
        multiples = np.arange(p, max_val + 1, p)
        # For each multiple, count how many times p divides it
        # We can do this iteratively
        remaining = multiples.copy()
        while len(remaining) > 0:
            sf[remaining] *= p
            # Find which of these are still divisible by p
            new_remaining = remaining[remaining % p == 0]
            remaining = new_remaining // p
    
    # Convert to list for easier slicing and set operations
    # Actually, let's keep it as numpy array and use numpy for window counting
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Iterate over valid n
    # We can optimize the window counting using numpy
    for n in range(1000, max_n + 1):
        # Check time periodically to avoid timeout
        if n % 10000 == 0:
            if time.time() > deadline:
                break
        
        # Extract factors for n+1 .. n+24
        window = sf[n + 1 : n + 25]
        
        # Count distinct values using numpy
        count = len(np.unique(window))
        
        if count < best_score:
            best_score = count
            best_n = n
            if best_score == 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
