import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    min_n = 10000
    
    # 1. Compute largest prime factor for all numbers up to N
    # Using a sieve approach: for each prime i, set lpf[j] = i for all multiples j
    # This correctly identifies the LARGEST prime factor because we iterate i in increasing order
    # and overwrite lpf[j] with larger primes as we go.
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
        if i % 100000 == 0 and time.time() > deadline:
            break

    # 2. Find n in [10000, N) minimizing max(lpf[n], lpf[n+1]) using NumPy
    lpf_arr = np.array(lpf, dtype=np.int32)
    vals = np.maximum(lpf_arr[min_n:N], lpf_arr[min_n+1:N+1])
    best_idx = int(np.argmin(vals))
    best_n = min_n + best_idx
    
    return best_n
