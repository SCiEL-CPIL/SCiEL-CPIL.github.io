"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    min_n = 1000
    max_n = 1000000
    
    # We need factors for numbers in [1001, 1000024]
    start_num = min_n + 1
    end_num = max_n + 24
    
    # Generate array of numbers
    nums = np.arange(start_num, end_num + 1, dtype=np.int64)
    
    # Initialize factors array
    fac = np.ones(len(nums), dtype=np.int64)
    
    # Compute small-prime factors for all numbers (vectorized)
    for p in primes:
        mask = (nums % p == 0)
        fac[mask] *= p
    
    # Now fac[i] corresponds to the factor of nums[i]
    # nums[0] = 1001, which is n+1 for n=1000
    # So for a candidate n, let k = n - 1000
    # The factors are fac[k : k+24]
    
    best_n = min_n
    best_score = 100  # Initialize with a high value
    
    # Use sliding window with a count map for efficiency
    cnt_map = defaultdict(int)
    
    # Initialize window for n=1000: indices 0 to 23 in fac
    for i in range(24):
        v = fac[i]
        cnt_map[v] += 1
    cur_distinct = len(cnt_map)
    if cur_distinct < best_score:
        best_score = cur_distinct
        best_n = min_n
    
    # Slide the window
    max_k = max_n - min_n  # 999000
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        
        # Remove fac[k-1], add fac[k+23]
        vout = fac[k-1]
        cnt_map[vout] -= 1
        if cnt_map[vout] == 0:
            del cnt_map[vout]
        
        vin = fac[k+23]
        cnt_map[vin] = cnt_map.get(vin, 0) + 1
        
        cur_distinct = len(cnt_map)
        if cur_distinct < best_score:
            best_score = cur_distinct
            best_n = k + min_n
            if best_score <= 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
