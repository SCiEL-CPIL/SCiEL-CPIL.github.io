import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80  # Safety margin for 90s limit
    
    N = 1000000
    max_val = N + 1  # We need lpf for N+1 as well
    
    # Sieve to find Largest Prime Factor using NumPy for speed
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # For each prime i, set lpf[j] = i for all multiples j of i.
    # Since we iterate i in increasing order, the last assignment wins,
    # giving us the largest prime factor.
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime - use numpy slicing for speed
            lpf[i::i] = i
            
        # Time check every 10000 iterations to avoid overhead
        if i % 10000 == 0 and time.time() - start_time > deadline:
            break
    
    # Check time
    if time.time() - start_time > deadline:
        return 10000  # Fallback, unlikely to happen
        
    min_lpf = float('inf')
    best_n = 10000
    
    # Convert to list for faster indexing in the scan loop
    lpf_list = lpf.tolist()
    
    # Search for n in [10000, 1000000]
    for n in range(10000, N + 1):
        # Calculate largest prime factor of n(n+1)
        current_max = lpf_list[n]
        next_max = lpf_list[n + 1]
        if next_max > current_max:
            current_max = next_max
            
        if current_max < min_lpf:
            min_lpf = current_max
            best_n = n
            
    return best_n
