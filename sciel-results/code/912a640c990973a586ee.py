"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

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
    
    # Precompute smooth part for each number using efficient sieve
    # smooth[x] = product of p^v_p(x) for all primes p < 24
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        # For multiples of p, compute p^v_p(multiple)
        # Use vectorized approach: for each power p^k, mark multiples
        power = p
        while power <= max_val:
            # All multiples of power but not of power*p have exactly v_p = k
            # So for m in range(power, max_val+1, power):
            #   if m % (power*p) != 0: smooth[m] *= power
            # But simpler: just multiply power into all multiples of power
            # and handle the overcounting by using the fact that
            # v_p(m) = number of times p divides m
            # Alternative: for each multiple of p, compute v_p and multiply p^v_p
            # More efficient: use the sieve property
            for m in range(power, max_val + 1, power):
                if m % (power * p) != 0:
                    smooth[m] *= power
            power *= p
            if power > max_val:
                break
    
    # Now slide window: for n in [1000, 1000000], look at smooth[n+1]..smooth[n+24]
    # Count distinct values in that window using sliding window frequency
    best_n = 1000
    best_count = float('inf')
    
    n_start = 1000
    n_end = 1000000
    
    # Initialize frequency dictionary for the first window
    freq = defaultdict(int)
    distinct_count = 0
    
    # First window: n = n_start, indices n_start+1 to n_start+24
    for i in range(n_start + 1, n_start + 25):
        val = int(smooth[i])
        freq[val] += 1
        if freq[val] == 1:
            distinct_count += 1
    
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = n_start
    
    # Slide the window with periodic time checks
    check_interval = 10000
    for n in range(n_start + 1, n_end + 1):
        # When we move from window starting at n-1 to n:
        # Remove smooth[n] (the leftmost element of previous window)
        # Add smooth[n+24] (the new rightmost element)
        
        # Remove smooth[n]
        val_out = int(smooth[n])
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            del freq[val_out]
        
        # Add smooth[n+24]
        val_in = int(smooth[n + 24])
        if freq.get(val_in, 0) == 0:
            distinct_count += 1
        freq[val_in] = freq.get(val_in, 0) + 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 2:
                break
        
        if (n - n_start) % check_interval == 0 and time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
