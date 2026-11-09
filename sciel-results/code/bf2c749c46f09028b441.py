import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80  # Safety margin for 90s limit
    
    N = 1000000
    max_val = N + 1  # We need lpf for N+1 as well
    
    # Use NumPy for faster sieve operations
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
            
    # Check time
    if time.time() - start_time > deadline:
        return 10000 # Fallback, unlikely to happen
        
    min_lpf = float('inf')
    best_n = 10000
    
    # Convert to list for faster scalar access in the tight scan loop?
    # Or just use numpy. List access is faster in Python.
    lpf_list = lpf.tolist()
    
    for n in range(10000, N + 1):
        current_max = lpf_list[n]
        next_max = lpf_list[n+1]
        if next_max > current_max:
            current_max = next_max
            
        if current_max < min_lpf:
            min_lpf = current_max
            best_n = n
            
    return best_n
