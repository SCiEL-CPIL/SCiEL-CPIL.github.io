import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    # Using a list of integers is fast enough for 1M elements
    extracted = [1] * (MAX_VAL + 1)
    
    for p in small_primes:
        # For each multiple of p, calculate the power of p dividing it
        # and multiply it into extracted[m]
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # Sliding window approach to count distinct extracted values
    # Window for n is [n+1, n+24]
    
    # Initialize window for n = N_MIN
    window_counts = defaultdict(int)
    for i in range(1, WINDOW + 1):
        window_counts[extracted[N_MIN + i]] += 1
    
    best_n = N_MIN
    best_count = len(window_counts)
    
    # Iterate n from N_MIN + 1 to N_MAX
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # The window for n-1 was [n, n+23]
        # The window for n is [n+1, n+24]
        # So we remove extracted[n] and add extracted[n+24]
        
        rem_val = extracted[n]
        add_val = extracted[n + 24]
        
        window_counts[rem_val] -= 1
        if window_counts[rem_val] == 0:
            del window_counts[rem_val]
        
        window_counts[add_val] += 1
        
        current_distinct = len(window_counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                # Theoretical minimum, cannot do better
                break

    return best_n
