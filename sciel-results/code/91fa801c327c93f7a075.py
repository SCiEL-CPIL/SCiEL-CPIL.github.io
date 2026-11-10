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
    # Use a sieve-like approach for speed: for each small prime p,
    # multiply p into small_factor[x] for all x divisible by p^k
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each power p^k, multiply p into all multiples of p^k
        power = p
        while power <= max_val:
            # Indices: power, 2*power, 3*power, ...
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
            
    # The values in small_factor are <= max_val (since they divide the number)
    # So we can use an array of size max_val + 1 for frequencies
    freq = np.zeros(max_val + 1, dtype=np.int32)
    
    # Initialize the first window for n=1000
    # Window covers indices 1001 to 1024
    start_idx = 1001
    end_idx = 1025 # exclusive
    
    distinct_count = 0
    for i in range(start_idx, end_idx):
        val = small_factor[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
        
    best_score = distinct_count
    best_n = 1000
    
    if best_score == 1:
        return 1000
        
    # Slide the window
    # n goes from 1001 to 999999
    # Window for n covers [n+1, n+24]
    # Previous window covered [n, n+23]
    # Remove n, Add n+24
    # Wait, let's trace carefully.
    # n=1000: window [1001, 1024]
    # n=1001: window [1002, 1025]
    # Remove index 1001, Add index 1025
    
    for n in range(1001, max_n + 1):
        if n % 10000 == 0 and time.time() > deadline:
            break
            
        # Remove element leaving the window: index (n-1)+1 = n
        # The window for n-1 was [(n-1)+1, (n-1)+24] = [n, n+23]
        # So we remove small_factor[n]
        val_out = small_factor[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            
        # Add element entering the window: index n+24
        # The window for n is [n+1, n+24]
        val_in = small_factor[n + 24]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
# EVOLVE-BLOCK-END
