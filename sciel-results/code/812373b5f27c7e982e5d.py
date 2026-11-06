import math
import time
import numpy as np

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    DEADLINE = 85  # safety margin before 90s kill
    
    start_time = time.time()
    
    # Sieve for Smallest Prime Factor
    max_val = N_MAX + 1
    spf = list(range(max_val + 1))
    
    # Standard SPF sieve
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all values 1..max_val
    lpf = [1] * (max_val + 1)
    for i in range(2, max_val + 1):
        x = i
        lp = 1
        while x > 1:
            p = spf[x]
            lp = p
            while x % p == 0:
                x //= p
        lpf[i] = lp
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return N_MIN
    
    # Vectorized scan using NumPy
    lpf_arr = np.array(lpf, dtype=np.int64)
    a = lpf_arr[N_MIN:N_MAX + 1]       # lpf[n] for n in [N_MIN, N_MAX]
    b = lpf_arr[N_MIN + 1:N_MAX + 2]   # lpf[n+1] for n in [N_MIN, N_MAX]
    vals = np.maximum(a, b)
    best_idx = int(np.argmin(vals))
    best_n = N_MIN + best_idx
    
    return best_n
