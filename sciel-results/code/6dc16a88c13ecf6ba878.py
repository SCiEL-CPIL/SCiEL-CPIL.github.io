"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    min_n = 1000
    max_val = max_n + 24  # 1000024
    min_val = min_n + 1   # 1001
    
    # Precompute small_prime_part for all m in [1, max_val]
    # Use NumPy for vectorized computation
    
    # Initialize rough_part as array of range
    rough_part = np.arange(max_val + 1, dtype=np.int64)
    
    # For each small prime, divide out all factors
    for p in small_primes:
        # Get indices of multiples of p
        multiples = np.arange(p, max_val + 1, p)
        # Divide out p repeatedly until no more factors
        # Use iterative vectorized operations
        for _ in range(20):  # 2^20 > 1e6, so 20 iterations is enough for any number
            mask = rough_part[multiples] % p == 0
            if not np.any(mask):
                break
            idx = multiples[mask]
            rough_part[idx] //= p
                
    # Now compute small_prime_part
    # We only need indices from min_val to max_val
    small_part = np.zeros(max_val + 1, dtype=np.int64)
    idx_range = np.arange(min_val, max_val + 1, dtype=np.int64)
    small_part[min_val:] = idx_range // rough_part[min_val:]
    
    # Convert to Python list for faster access in the sliding window loop
    small_part_list = small_part.tolist()
    
    # Search for best n
    # Sliding window of size 24
    # Window covers n+1 to n+24
    
    best_n = min_n
    best_count = 25 # Max possible is 24, so 25 is safe initial
    
    # Initialize frequency counter for the first window [1001, 1024]
    freq = defaultdict(int)
    distinct_count = 0
    
    for i in range(1, 25):
        val = small_part_list[min_n + i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
        
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = min_n
        
    # Slide the window
    # n goes from min_n + 1 to max_n
    # Window for n is [n+1, n+24]
    # Previous window for n-1 was [n, n+23]
    # Remove n, add n+24
    
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline:
            break
            
        # Remove element at n (which was n-1 + 1)
        val_out = small_part_list[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            
        # Add element at n+24 (which is n + 24)
        val_in = small_part_list[n + 24]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
            # If we reach the theoretical minimum of 1, we can stop
            if best_count == 1:
                break
                
    return best_n
# EVOLVE-BLOCK-END
