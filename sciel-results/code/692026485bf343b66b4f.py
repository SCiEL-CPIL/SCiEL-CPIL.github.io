def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 1000000
    limit = N + 1  # We need lpf up to N+1
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Largest Prime Factor Sieve using NumPy
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        lpf[p::p] = p
    
    # 3. Vectorized search for best n
    # lpf[n] for n in [10000, N] and lpf[n+1] for n in [10000, N]
    lpf_n = lpf[10000:N+1]      # LPF of n for n=10000..1000000
    lpf_np1 = lpf[10001:N+2]    # LPF of n+1 for n=10000..1000000
    
    scores = np.maximum(lpf_n, lpf_np1)
    best_idx = np.argmin(scores)
    best_n = 10000 + int(best_idx)
    
    return best_n
