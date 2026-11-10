import numpy as np
from collections import Counter

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    window_size = 24
    max_val = max_n + window_size
    
    # Precompute masks
    # masks[i] is a bitmask of primes in 'primes' dividing i
    masks = np.zeros(max_val + 1, dtype=np.int32)
    
    for k, p in enumerate(primes):
        # Set bit k for all multiples of p
        # Using numpy slicing for speed
        masks[p::p] |= (1 << k)
        
    # Sliding window
    # Initial window for n=1000: indices 1001 to 1024
    # We need to count distinct masks in masks[1001:1025]
    
    current_count = Counter(masks[1001:1025].tolist())
    current_distinct = len(current_count)
    
    min_distinct = current_distinct
    best_n = 1000
    
    # Slide from n=1000 to n=999999
    # Next n is n+1
    # Leaving index: n+1
    # Entering index: n+1 + window_size = n + 25
    for n in range(1000, max_n):
        # n is the current start, we are moving to n+1
        # The window for n is [n+1, ..., n+24]
        # The window for n+1 is [n+2, ..., n+25]
        
        # Remove mask at index n+1
        out_mask = masks[n+1]
        current_count[out_mask] -= 1
        if current_count[out_mask] == 0:
            current_distinct -= 1
            
        # Add mask at index n+25
        in_mask = masks[n+25]
        current_count[in_mask] += 1
        if current_count[in_mask] == 1:
            current_distinct += 1
            
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n + 1
            
    return best_n
