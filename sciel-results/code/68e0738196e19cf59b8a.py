import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 60.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Structural change: represent each number's small-prime factor by a bitmask
    # where bit i is set if prime[i] divides the number.
    # Two numbers have the same small-prime factor iff they have the same bitmask.
    # The number of distinct small-prime factors in a window = number of distinct bitmasks.
    
    # Compute bitmask for each number 1..size
    # bitmask[i] = OR of (1 << j) for each prime j that divides i
    bitmask = np.zeros(size + 1, dtype=np.uint16)
    
    for j, p in enumerate(primes):
        # For each multiple of p, set bit j
        multiples = np.arange(p, size + 1, p)
        bitmask[multiples] |= (1 << j)
    
    # Now for each window [n+1, n+25], count distinct bitmasks.
    # Structural approach: use a sliding window with a frequency array of size 512.
    # This is much faster than a defaultdict because it's a fixed-size array.
    
    freq = np.zeros(512, dtype=np.int32)
    
    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1001, 1025):
        freq[bitmask[i]] += 1
    
    distinct = np.count_nonzero(freq)
    best_n = 1000
    best_count = distinct
    
    if best_count == 1:
        return best_n
    
    deadline = start_time + timeout - 5.0
    
    # Slide window from n=1001 to max_n
    for n in range(1001, max_n + 1):
        # Remove element at position n (which was n+1 in previous window)
        out_val = bitmask[n]
        freq[out_val] -= 1
        
        # Add element at position n+24
        in_val = bitmask[n + 24]
        freq[in_val] += 1
        
        distinct = np.count_nonzero(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break
        
        if n % 10000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
