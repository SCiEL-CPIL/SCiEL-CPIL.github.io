import numpy as np

def construct():
    N = 1000000
    limit = N + 1
    
    # Sieve to find primes and largest prime factors
    is_prime = np.ones(limit + 1, dtype=bool)
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    is_prime[0] = False
    is_prime[1] = False
    lpf[1] = 1
    
    for i in range(2, limit + 1):
        if is_prime[i]:
            # i is a prime
            # Update largest prime factor for all multiples of i
            # Since i is processed in increasing order, the last write
            # to a specific index will be by the largest prime factor.
            lpf[i::i] = i
            
            # Mark multiples as non-prime for the sieve
            # Start from i*i to avoid redundant marking
            if i * i <= limit:
                is_prime[i*i::i] = False

    # We need to find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Create array of n values
    n_vals = np.arange(10000, N + 1)
    
    # Get lpf values for n and n+1
    lpf_n = lpf[n_vals]
    lpf_np1 = lpf[n_vals + 1]
    
    # Compute the metric to minimize
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    best_n = n_vals[min_idx]
    
    return int(best_n)
