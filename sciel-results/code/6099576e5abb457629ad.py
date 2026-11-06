import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    LIMIT = 1000000
    N = LIMIT + 1
    
    # 1. Sieve for primes up to sqrt(N)
    sqrt_N = int(N**0.5) + 1
    is_prime = [True] * (sqrt_N + 1)
    if sqrt_N >= 0:
        is_prime[0] = False
    if sqrt_N >= 1:
        is_prime[1] = False
    for i in range(2, int(sqrt_N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sqrt_N + 1) if is_prime[i]]
    
    # 2. Compute powerful part using NumPy sieve
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        return 10000
    
    indices = np.arange(N + 1, dtype=np.int64)
    pp_np = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        pe = p * p
        while pe <= N:
            pe_next = pe * p
            if pe_next <= N:
                mask = (indices % pe == 0) & (indices % pe_next != 0)
            else:
                mask = (indices % pe == 0)
            pp_np[mask] *= pe
            pe = pe_next
    
    # Check time before conversion and scan
    if time.time() - start_time > DEADLINE - 5:
        return 10000
    
    # 3. Find n that maximizes pp[n] * pp[n+1] using vectorized NumPy
    # n ranges from 10000 to LIMIT (1000000), so n+1 ranges from 10001 to LIMIT+1 (1000001)
    products = pp_np[10000:LIMIT + 1] * pp_np[10001:LIMIT + 2]
    best_idx = np.argmax(products)
    best_n = 10000 + int(best_idx)
    
    return best_n
