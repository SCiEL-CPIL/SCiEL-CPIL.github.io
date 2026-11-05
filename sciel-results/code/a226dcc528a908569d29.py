import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF) using NumPy
    spf = np.arange(N + 2, dtype=np.int64)
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            # Mark multiples: only set if spf[j] is still j (i.e., not yet assigned)
            start = i * i
            step = i
            # Use a loop for clarity and correctness
            for j in range(start, N + 2, step):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute powerful part for all numbers up to N
    # powerful[i] stores the powerful part of i
    powerful = np.ones(N + 2, dtype=np.int64)
    
    for k in range(2, N + 2):
        p = int(spf[k])
        m = k
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        
        if e >= 2:
            powerful[k] = powerful[m] * (p ** e)
        else:
            powerful[k] = powerful[m]
            
    # Check time
    if time.time() > deadline:
        return MIN_N  # Fallback, shouldn't happen

    # 3. Vectorized scan for max product
    # n ranges from MIN_N to N-1
    n_vals = np.arange(MIN_N, N, dtype=np.int64)
    products = powerful[n_vals] * powerful[n_vals + 1]
    best_idx = int(np.argmax(products))
    best_n = int(n_vals[best_idx])
    
    return best_n
