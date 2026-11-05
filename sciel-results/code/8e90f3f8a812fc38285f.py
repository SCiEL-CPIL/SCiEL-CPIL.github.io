import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    MAX_N = 1000000
    # We need smooth parts for n+1 to n+24, so up to MAX_N + 24
    max_val = MAX_N + 24
    # Allocate array for smooth parts
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute smooth parts using vectorized operations
    for p in primes:
        power = p
        while power <= max_val:
            # For each multiple of power, multiply the smooth part by p
            # This effectively adds one factor of p to the smooth part for all numbers divisible by power
            # Note: This logic works because if a number is divisible by p^k, it is divisible by p, p^2, ..., p^k.
            # The loop multiplies by p for each power p^k.
            # Example: x=12. p=2.
            # power=2: 12 is multiple of 2. smooth[12] *= 2.
            # power=4: 12 is multiple of 4. smooth[12] *= 2.
            # power=8: 12 is not multiple of 8.
            # Result: smooth[12] has factor 2*2=4. Correct.
            smooth[power::power] *= p
            power *= p
            
    best_n = 1000
    best_count = 25  # Max possible distinct factors is 24, but we start high
    
    # To speed up the loop, we can try to use a faster distinct count method.
    # np.unique is generally fast for small arrays.
    # We iterate through all n in range.
    
    # Optimization: Convert smooth to a list? No, numpy slicing is fast.
    # Optimization: Check time periodically.
    
    for n in range(1000, MAX_N + 1):
        # Check time every 10000 iterations to reduce overhead
        if (n - 1000) % 10000 == 0:
            if time.time() > deadline:
                break
                
        # Extract the 24 values
        # Slicing numpy array creates a view
        window = smooth[n+1:n+25]
        
        # Count distinct values
        # np.unique sorts the array and removes duplicates.
        # For 24 elements, this is very fast.
        count = len(np.unique(window))
        
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                # Cannot do better than 1
                break
                
    return best_n
