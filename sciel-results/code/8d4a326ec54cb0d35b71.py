import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Sieve-based computation of small-prime factors
    # small_factor[x] = product of p^v_p(x) for all small primes p
    small_factor = [1] * (max_val + 1)
    
    for p in small_primes:
        pk = p
        while pk <= max_val:
            # Use local variable for speed
            sf = small_factor
            for multiple in range(pk, max_val + 1, pk):
                sf[multiple] *= p
            pk *= p

    # Sliding window scan with frequency array
    # Values of small_factor are <= max_val, so we can use a list of size max_val + 1
    freq_arr = [0] * (max_val + 1)
    active_count = 0
    
    # Initialize for n=1000: window is [1001, 1024]
    for i in range(1, 25):
        val = small_factor[1000 + i]
        if freq_arr[val] == 0:
            active_count += 1
        freq_arr[val] += 1
        
    best_score = active_count
    best_n = 1000
    
    if best_score <= 2:
        return best_n

    # Slide through all windows
    check_interval = 10000
    fa = freq_arr # local ref
    sf = small_factor
    for n in range(1001, max_n + 1):
        if (n - 1001) % check_interval == 0:
            if time.time() > deadline:
                break
        
        old_val = sf[n]
        fa[old_val] -= 1
        if fa[old_val] == 0:
            active_count -= 1
            
        new_val = sf[n + 24]
        if fa[new_val] == 0:
            active_count += 1
        fa[new_val] += 1
        
        if active_count < best_score:
            best_score = active_count
            best_n = n
            if best_score <= 2:
                break
            # If we found a very good score, we can be more aggressive with time checks
            if best_score <= 3:
                check_interval = 1000

    return best_n
