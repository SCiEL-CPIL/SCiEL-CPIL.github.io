import numpy as np

def construct():
    N = 1000000
    lpf = np.zeros(N + 2, dtype=np.int32)
    
    # Find primes using a boolean sieve
    is_prime = np.ones(N + 2, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(np.sqrt(N + 1)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Get list of primes
    primes = np.where(is_prime)[0]
    
    # Update lpf for each prime
    # We must iterate in increasing order of primes to ensure largest prime factor is stored
    for p in primes:
        # p is prime
        # Update multiples of p
        # lpf[j] = p for j in multiples of p
        # This overwrites previous values with larger primes
        lpf[p::p] = p
        
    best_n = 10000
    best_max_pf = float('inf')
    
    n_range = np.arange(10000, N + 1)
    pf_n = lpf[n_range]
    pf_np1 = lpf[n_range + 1]
    
    max_pfs = np.maximum(pf_n, pf_np1)
    
    min_idx = np.argmin(max_pfs)
    best_n = int(n_range[min_idx])
    
    return best_n
