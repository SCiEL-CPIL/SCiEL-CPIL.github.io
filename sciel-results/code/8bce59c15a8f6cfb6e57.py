# EVOLVE-BLOCK-START
import numpy as np

def construct():
    max_n = 1000000
    window_size = 24
    max_val = max_n + window_size + 10 # Safety margin
    
    # Small primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_small_primes = len(small_primes)
    
    # masks[m] will have bit i set if small_primes[i] divides m
    masks = np.zeros(max_val + 1, dtype=np.int32)
    
    for i, p in enumerate(small_primes):
        bit = 1 << i
        # Set bit for all multiples of p
        # We can use vectorized assignment
        multiples = np.arange(p, max_val + 1, p)
        masks[multiples] |= bit
        
    # Sliding window
    # bit_counts[j] = number of numbers in current window with bit j set
    bit_counts = np.zeros(num_small_primes, dtype=np.int32)
    current_distinct = 0
    
    # Helper to add a mask to the window state
    def add_mask(m):
        nonlocal current_distinct
        for j in range(num_small_primes):
            if (m >> j) & 1:
                if bit_counts[j] == 0:
                    current_distinct += 1
                bit_counts[j] += 1
                
    def remove_mask(m):
        nonlocal current_distinct
        for j in range(num_small_primes):
            if (m >> j) & 1:
                bit_counts[j] -= 1
                if bit_counts[j] == 0:
                    current_distinct -= 1
                    
    # Initialize window for n=1000 (indices 1001 to 1024)
    for i in range(1001, 1001 + window_size):
        add_mask(masks[i])
        
    best_n = 1000
    best_count = current_distinct
    
    # Slide window
    # n goes from 1000 to max_n
    # Window for n is [n+1, n+24]
    # Next n is n+1, window is [n+2, n+25]
    # Remove n+1, add n+25
    for n in range(1000, max_n):
        # Remove n+1
        remove_mask(masks[n+1])
        # Add n+25
        add_mask(masks[n+25])
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n + 1 # The n for the current window is n+1? 
                           # Wait.
                           # Initial n=1000, window [1001, 1024].
                           # Loop n=1000: remove 1001, add 1025.
                           # New window is [1002, 1025]. This corresponds to n=1001.
                           # So best_n should be updated to n+1.
                           
                           # Let's re-verify.
                           # We want to return n such that window is [n+1, n+24].
                           # Initial: n=1000.
                           # After 1st iteration of loop (n=1000): window is [1002, 1025]. This is for n=1001.
                           # So yes, best_n = n+1.
                           
            if best_count == 0:
                break # Can't do better than 0? No, min is 1? 
                      # Actually, if no primes < 24 divide any number in the window, count is 0.
                      # But 24 consecutive integers must contain multiples of 2, 3, etc.
                      # So count will be at least a few.
                      
    return best_n

# EVOLVE-BLOCK-END
