import sys
import time
import numpy as np
from collections import defaultdict

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
    
    sp = small_part
    deadline = start_time + timeout - 5.0
    
    # Sliding window with frequency map for O(1) amortized distinct count
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: indices 1001 to 1024
    freq = defaultdict(int)
    for i in range(1001, 1025):
        val = int(sp[i])
        freq[val] += 1
    
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    if best_count == 1:
        return best_n
    
    # Slide window from n=1001 to max_n
    for n in range(1001, max_n + 1):
        # Remove element at position n (which was n+1 in previous window)
        out_val = int(sp[n])
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
        
        # Add element at position n+24
        in_val = int(sp[n + 24])
        freq[in_val] += 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break
        
        if n % 50000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
