import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    window_size = 24
    max_val = max_n + window_size
    
    # Precompute masks
    masks = np.zeros(max_val + 1, dtype=np.uint8)
    for j, p in enumerate(primes):
        masks[p::p] |= (1 << j)
        
    # Initialize counts for first window n=1000
    # Window covers 1001 to 1024
    bit_counts = [0] * len(primes)
    current_or = 0
    
    # Helper to update counts and current_or
    def add_mask(m):
        nonlocal current_or
        # Iterate through bits set in m
        # Since m is small, we can just loop j
        for j in range(len(primes)):
            if m & (1 << j):
                bit_counts[j] += 1
                if bit_counts[j] == 1:
                    current_or |= (1 << j)
                    
    def remove_mask(m):
        nonlocal current_or
        for j in range(len(primes)):
            if m & (1 << j):
                bit_counts[j] -= 1
                if bit_counts[j] == 0:
                    current_or &= ~(1 << j)

    # Init
    for i in range(1, window_size + 1):
        add_mask(masks[1000 + i])
        
    best_n = 1000
    best_val = bin(current_or).count('1')
    
    # Slide
    for n in range(1001, max_n + 1):
        if time.time() > deadline:
            break
            
        # Remove n, add n+24
        remove_mask(masks[n])
        add_mask(masks[n + window_size])
        
        val = bin(current_or).count('1')
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
