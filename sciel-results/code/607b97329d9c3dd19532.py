import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 1000002
    
    # Sieve for primes
    is_prime = np.ones(N, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:N:i] = False
            
    # Compute Largest Prime Factor
    lpf = np.zeros(N, dtype=np.int64)
    for p in range(2, N):
        if is_prime[p]:
            lpf[p:N:p] = p
            
    # Search for min max(lpf[n], lpf[n+1])
    # We need to evaluate max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # Let's create arrays for the range
    ns = np.arange(10000, 1000001)
    vals = np.maximum(lpf[ns], lpf[ns+1])
    
    min_val = np.min(vals)
    min_idx = np.argmin(vals)
    best_n = ns[min_idx]
    
    return int(best_n)
