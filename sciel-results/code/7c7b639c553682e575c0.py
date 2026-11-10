"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 85
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    lo = 1001
    hi = 1000024
    size = hi - lo + 1
    
    # Precompute small_part for each number in [lo, hi]
    # small_part[m] is the part of m composed of primes < 24
    # Note: indices in array 0..size-1 correspond to numbers lo..hi
    # We need small_part for n+1..n+24 where n in [1000, 1000000]
    # So we need numbers 1001..1000024
    
    # Create array of numbers
    nums = np.arange(lo, hi + 1, dtype=np.int64)
    
    # For each prime, divide out all factors
    for p in primes:
        # We need to divide by p as many times as possible
        # This is equivalent to: while num % p == 0: num //= p
        # Vectorized approach:
        # mask = nums % p == 0
        # nums[mask] //= p
        # Repeat until no more
        # Since max value is 10^6, max power of 2 is 2^19, so ~19 iterations max for p=2.
        # For p=23, max power is 23^4 < 10^6, so 4 iterations.
        # Total iterations small.
        changed = True
        while changed:
            mask = nums % p == 0
            if not np.any(mask):
                changed = False
            else:
                nums[mask] //= p
    
    # Now nums[i] is the small part of (lo + i)
    # We slide a window of 24 over nums
    # Window starting at index i (0-based) covers nums[i]..nums[i+23]
    # This corresponds to n = lo - 1 + i
    # Valid n in [1000, 1000000]
    # lo = 1001, so n = 1000 + i
    # i ranges from 0 to 999000
    
    max_i = 999000
    
    best_count = 100
    best_n = 1000
    
    # Initialize counts for window at i=0
    # Window covers nums[0]..nums[23]
    counts = {}
    for j in range(24):
        val = nums[j]
        counts[val] = counts.get(val, 0) + 1
    
    current_distinct = len(counts)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000
        
    # Slide window
    for i in range(1, max_i + 1):
        # Remove nums[i-1]
        val_out = nums[i-1]
        counts[val_out] -= 1
        if counts[val_out] == 0:
            del counts[val_out]
            
        # Add nums[i+23]
        val_in = nums[i+23]
        counts[val_in] = counts.get(val_in, 0) + 1
        
        current_distinct = len(counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = 1000 + i
            
        # Time check
        if time.time() > deadline:
            break
            
    return best_n
# EVOLVE-BLOCK-END
