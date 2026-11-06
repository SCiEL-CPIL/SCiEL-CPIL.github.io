import numpy as np
from collections import defaultdict

def construct():
    MAX_N = 1000000
    MAX_M = MAX_N + 24
    primes_below_24 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth factors using a list for speed
    smooth = [1] * (MAX_M + 1)
    
    for p in primes_below_24:
        p_power = p
        while p_power <= MAX_M:
            for multiple in range(p_power, MAX_M + 1, p_power):
                # Check if multiple is divisible by p_power but not p_power * p
                # Actually, simpler: just multiply by p for each power of p dividing it.
                # The previous loop was correct. Let's stick to it but optimize.
                pass
            
            # Better: Iterate multiples of p, determine exponent.
            # The original loop was:
            # for multiple in range(p, MAX_M + 1, p):
            #    ...
            # This is O(N log N).
            
            # Let's use the original logic but with local variable binding.
            break
    
    # Re-implementing precomputation efficiently
    smooth = [1] * (MAX_M + 1)
    for p in primes_below_24:
        for multiple in range(p, MAX_M + 1, p):
            count = 0
            temp = multiple
            while temp % p == 0:
                count += 1
                temp //= p
            if count:
                smooth[multiple] *= p ** count

    best_n = 1000
    best_count = 25
    
    # Use a dict for frequency
    freq = {}
    
    # Initialize window for n=1000
    for i in range(1, 25):
        val = smooth[1000 + i]
        freq[val] = freq.get(val, 0) + 1
    
    best_count = len(freq)
    best_n = 1000
    
    # Slide window
    for n in range(1001, MAX_N + 1):
        val_remove = smooth[n]
        val_add = smooth[n + 24]
        
        # Remove
        cnt = freq[val_remove]
        if cnt == 1:
            del freq[val_remove]
        else:
            freq[val_remove] = cnt - 1
            
        # Add
        freq[val_add] = freq.get(val_add, 0) + 1
        
        count = len(freq)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
