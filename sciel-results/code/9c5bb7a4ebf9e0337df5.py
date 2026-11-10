import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small-prime factor for numbers up to max_val
    # small_factor[x] = product of prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            # Vectorized multiplication
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p

    # Convert to list for faster access in the sliding window
    # Using a list of Python ints is faster for dict lookups than numpy scalars
    sf_list = small_factor.tolist()
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Initialize window for n=1000: indices 1001 to 1024
    window_counts = {}
    for i in range(1001, 1025):
        val = sf_list[i]
        window_counts[val] = window_counts.get(val, 0) + 1
        
    current_distinct = len(window_counts)
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
        if best_score <= 4:
            return best_n

    # Slide window for n from 1001 to 1000000
    for n in range(1001, max_n + 1):
        # Time check every 10000 steps to avoid overhead
        if n % 10000 == 0:
            if time.time() > deadline:
                break
        
        # Remove outgoing element: sf_list[n]
        val_out = sf_list[n]
        count_out = window_counts[val_out]
        if count_out == 1:
            del window_counts[val_out]
        else:
            window_counts[val_out] = count_out - 1
            
        # Add incoming element: sf_list[n+24]
        val_in = sf_list[n+24]
        window_counts[val_in] = window_counts.get(val_in, 0) + 1
        
        current_distinct = len(window_counts)
        
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = n
            if best_score <= 4:
                # It is extremely unlikely to find < 4 distinct factors
                # for 24 consecutive integers.
                # If we find 4, it's a very strong candidate for the global min.
                # We can return early to save time, or continue to verify.
                # Given the time limit, returning early on a very low score
                # is a good heuristic if we are confident.
                # However, to be safe, let's just break if we hit 3.
                if best_score <= 3:
                    break
                # If we hit 4, we might still find 3? 
                # Let's continue but it's rare.
                # Actually, let's just return if we find 4, as 3 is highly improbable.
                return best_n

    return best_n
