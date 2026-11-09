import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer for safety
    
    N = 1000001
    limit_sqrt = int(N**0.5) + 1
    
    # Sieve primes up to limit_sqrt using NumPy
    is_prime = np.ones(limit_sqrt + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit_sqrt**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Compute Powerful Part P(n) for all n using NumPy vectorization
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        p2 = p * p
        if p2 > N:
            break
        pe = p2
        while pe <= N:
            pe_next = pe * p
            if pe_next > N:
                P[pe::pe] *= pe
                break
            else:
                P[pe::pe] *= pe
                P[pe_next::pe_next] //= pe
                pe = pe_next
        
        if time.time() > deadline:
            break
    
    # Find max P[n]*P[n+1] for n in [10000, 1000000]
    scores = P[10000:1000000] * P[10001:1000001]
    best_idx = int(np.argmax(scores))
    best_n = 10000 + best_idx
    
    return best_n
