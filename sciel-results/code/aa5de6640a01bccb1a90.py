import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    MAX_N = 1000000
    MIN_N = 10000
    SIZE = MAX_N + 1  # need indices up to 1000001
    
    # Sieve for largest prime factor: O(n log log n)
    lpf = np.zeros(SIZE + 1, dtype=np.int64)
    for i in range(2, SIZE + 1):
        if lpf[i] == 0:
            for j in range(i, SIZE + 1, i):
                lpf[j] = i  # Since we iterate i in increasing order, the last prime divisor is the largest
    
    # Scan for best consecutive pair: O(n) with O(1) lookups
    best_n = MIN_N
    best_val = float('inf')
    
    for n in range(MIN_N, MAX_N + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline - 2:
            break
    
    return int(best_n)
