"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 82  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    limit = max_n + 25
    
    # Compute 23-smooth part for each number from 1 to limit
    # smooth_part[m] = product of p^v_p(m) for p in primes (primes < 24)
    sp = np.ones(limit + 1, dtype=np.int64)
    
    # For each small prime, multiply sp[m] by p^v_p(m)
    # v_p(m) = exponent of p in m
    # We can compute this by repeatedly dividing
    
    # Efficient approach: for each prime p, for each power p^k, 
    # for each multiple of p^k, we add one factor of p to the smooth part
    # But we need to be careful about overcounting
    
    # Better: for each number m, compute smooth part directly
    # But vectorize using the fact that v_p(m) can be computed via repeated division
    
    # Alternative efficient method:
    # For each prime p, create an array of v_p(m) for all m
    # Then smooth_part = product of p^v_p(m)
    
    # Let's use a different approach: 
    # sp[m] = m
    # For each prime p >= 24, divide sp[m] by p as many times as possible for each m divisible by p
    # But we need to know which primes are >= 24
    
    # Fastest approach for 1M numbers: direct trial division by 9 primes
    # For each m from 1 to limit:
    #   temp = m
    #   result = 1
    #   for p in primes:
    #       while temp % p == 0:
    #           result *= p
    #           temp //= p
    #   sp[m] = result
    # This is O(limit * 9 * log_p(m)) ~ 1M * 9 * ~5 = 45M operations, should be fine
    
    # But Python loops are slow. Let's use numpy where possible.
    # Actually, let's just use a Python loop but optimize it
    
    sp = np.zeros(limit + 1, dtype=np.int64)
    sp[0] = 0
    sp[1] = 1
    
    for m in range(2, limit + 1):
        temp = m
        result = 1
        for p in primes:
            while temp % p == 0:
                result *= p
                temp //= p
        sp[m] = result
    
    # Now find the window with minimum distinct smooth parts
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window with Counter
    from collections import Counter
    count_dict = Counter()
    
    # Initialize window for n=1000: sp[1001] to sp[1024]
    for i in range(1001, 1025):
        count_dict[sp[i]] += 1
    
    distinct = len(count_dict)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide
    for n in range(1001, max_n + 1):
        old_val = sp[n]      # leaving the window
        new_val = sp[n + 24] # entering the window
        
        count_dict[old_val] -= 1
        if count_dict[old_val] == 0:
            del count_dict[old_val]
        
        count_dict[new_val] += 1
        
        distinct = len(count_dict)
        if distinct < best_count:
            best_count = distinct
            best_n = n
        
        if time.time() - start_time > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
