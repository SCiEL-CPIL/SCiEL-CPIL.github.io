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
    then slide a window of 24 and count distinct smooth parts.
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute smooth part for each number
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p, divide out all factors of p and track
        # the smooth contribution
        # Actually, we want: smooth_part[x] = product of p^k for each prime p<24
        # where p^k is the highest power of p dividing x
        # So smooth[x] = x / (x with all small prime factors removed)
        # Equivalently: smooth[x] = x * prod_{p<24} (1/p^v_p(x))... no
        # smooth[x] = prod_{p<24} p^{v_p(x)}
        
        # Start with all 1s, multiply in p^v_p(x) for each x
        for multiple in range(p, max_val + 1, p):
            power = 1
            temp = multiple
            while temp % p == 0:
                power *= p
                temp //= p
            smooth[multiple] *= power
    
    # Now slide window: for n in [1000, 1000000], look at smooth[n+1]..smooth[n+24]
    # Count distinct values in that window using sliding window frequency
    best_n = 1000
    best_count = float('inf')
    
    n_start = 1000
    n_end = 1000000
    
    # Initialize frequency dictionary for the first window
    from collections import defaultdict
    freq = defaultdict(int)
    distinct_count = 0
    
    # First window: n = n_start, indices n_start+1 to n_start+24
    for i in range(n_start + 1, n_start + 25):
        val = smooth[i]
        freq[val] += 1
        if freq[val] == 1:
            distinct_count += 1
    
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = n_start
    
    # Slide the window
    for n in range(n_start + 1, n_end + 1):
        # Remove element at position n (which is n_start + (n - n_start) - 1 + 1 = ... )
        # When we move from window starting at n-1 to n:
        # Remove smooth[n] (the leftmost element of previous window)
        # Add smooth[n+24] (the new rightmost element)
        
        # Remove smooth[n]
        val_out = smooth[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            del freq[val_out]
        
        # Add smooth[n+24]
        val_in = smooth[n + 24]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
        
        if (n - n_start) % 50000 == 0 and time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
