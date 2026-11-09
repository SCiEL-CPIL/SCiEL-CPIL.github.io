import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers up to 1000000 + 24 = 1000024
    max_val = 1000024
    sp = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve: for each small prime p, multiply sp[m] by p for each power p^k dividing m
    # This is equivalent to sp[m] *= p^v_p(m)
    for p in small_primes:
        p_power = p
        while p_power <= max_val:
            # Vectorized update: sp[m] *= p for all m divisible by p_power
            # Using modulo on a precomputed range is efficient
            # Note: np.arange(0, max_val+1) is created each time, which is overhead.
            # We can precompute indices or use slicing if possible, but modulo is straightforward.
            # Optimization: create the range once?
            # indices = np.arange(0, max_val + 1)
            # But memory is 10^6 * 8 bytes = 8MB. Fine.
            pass # Placeholder
            
        # Let's optimize the inner loop
        indices = np.arange(0, max_val + 1)
        p_power = p
        while p_power <= max_val:
            mask = indices % p_power == 0
            sp[mask] *= p
            p_power *= p
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Convert to list for faster iteration in the sliding window
    sp_list = sp.tolist()
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window over sp[n+1 : n+25] for n in [1000, 1000000]
    # Window for n: indices n+1 to n+24 (inclusive)
    
    # Initialize counter for n=1000: window [1001, 1024]
    freq = {}
    distinct_count = 0
    
    for i in range(1001, 1025):
        val = sp_list[i]
        if val not in freq:
            freq[val] = 1
            distinct_count += 1
        else:
            freq[val] += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
        
    if best_count == 1:
        return best_n

    # Slide window: for new_n from 1001 to 1000000
    for new_n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove leftmost (index new_n)
        old_val = sp_list[new_n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
            
        # Add new rightmost (index new_n + 24)
        new_val = sp_list[new_n + 24]
        if new_val not in freq:
            freq[new_val] = 1
            distinct_count += 1
        else:
            if freq[new_val] == 0:
                distinct_count += 1
            freq[new_val] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = new_n
            if best_count == 1:
                break
    
    return best_n
