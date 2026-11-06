def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 1000000
    limit = N + 1
    
    # 1. Sieve of Eratosthenes using NumPy
    # Use a bytearray or bool array
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Sieve
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Check time
    if time.time() > deadline:
        return 10000

    primes = np.where(is_prime)[0].tolist()
    
    # 2. Largest Prime Factor Sieve
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # Optimization: Only iterate primes up to limit
    # The loop assigns the largest prime factor because primes are in increasing order
    for p in primes:
        if p > limit:
            break
        lpf[p::p] = p
        
    # Check time
    if time.time() > deadline:
        return 10000

    # 3. Search for best n
    # We want to minimize max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    
    n_range = np.arange(10000, N + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    if len(scores) > 0:
        min_idx = np.argmin(scores)
        best_n = int(n_range[min_idx])
    else:
        best_n = 10000
        
    return best_n
