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
    
    # small_factor[x] = product of prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p

    sf_list = small_factor.tolist()
    
    best_score = 25
    best_n = 1000
    
    # Initialize window for n=1000: indices 1001 to 1024
    window_counts = {}
    for i in range(1001, 1025):
        v = sf_list[i]
        window_counts[v] = window_counts.get(v, 0) + 1
        
    current_distinct = len(window_counts)
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
        if best_score == 1:
            return best_n
    
    # Slide window for n from 1001 to 1000000
    for n in range(1001, max_n + 1):
        if n % 50000 == 0:
            if time.time() > deadline:
                break
        
        # Remove outgoing element: sf_list[n]
        val_out = sf_list[n]
        c = window_counts[val_out]
        if c == 1:
            del window_counts[val_out]
            current_distinct -= 1
        else:
            window_counts[val_out] = c - 1
            
        # Add incoming element: sf_list[n+24]
        val_in = sf_list[n + 24]
        c = window_counts.get(val_in, 0)
        if c == 0:
            window_counts[val_in] = 1
            current_distinct += 1
        else:
            window_counts[val_in] = c + 1
        
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = n
            if best_score == 1:
                break
    
    return best_n
