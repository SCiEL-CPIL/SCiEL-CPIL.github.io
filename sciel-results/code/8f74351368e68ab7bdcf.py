import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    LIMIT = 1000000
    MIN_N = 10000
    MAX_N = LIMIT
    SIZE = MAX_N + 2
    
    # Sieve for largest prime factor
    lpf = np.zeros(SIZE, dtype=np.int64)
    is_prime = np.ones(SIZE, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, SIZE):
        if is_prime[i]:
            # i is prime: mark multiples and set LPF
            if i * i < SIZE:
                is_prime[i * i : SIZE : i] = False
            lpf[i : SIZE : i] = i
        if time.time() > deadline - 5.0:
            break
    
    # Find n that minimizes max(lpf[n], lpf[n+1])
    lpf_n = lpf[MIN_N : MAX_N + 1]
    lpf_np1 = lpf[MIN_N + 1 : SIZE]
    max_lpf = np.maximum(lpf_n, lpf_np1)
    best_idx = np.argmin(max_lpf)
    best_n = MIN_N + best_idx
    
    return int(best_n)
