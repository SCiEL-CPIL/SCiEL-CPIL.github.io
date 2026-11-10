import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 60.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Vectorized sieve: for each prime p and each power p^k,
    # multiply small_part at indices divisible by p^k by p.
    # This correctly accumulates p^(v_p(n)) for each n.
    small_part = np.ones(size + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= size:
            small_part[power::power] *= p
            power *= p
    
    best_n = 1000
    best_count = float('inf')
    sp = small_part
    deadline = start_time + timeout - 5.0
    check_interval = 10000
    
    for n in range(1000, max_n + 1):
        window = sp[n+1 : n+25]
        count = len(set(window.tolist()))
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
        if (n - 1000) % check_interval == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
