import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 1000000
    # Initialize largest prime factor array
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(N + 2, dtype=np.int32)
    
    # Sieve to find largest prime factors
    # We iterate i from 2 to N. If lpf[i] is 0, i is prime.
    # For a prime i, we update lpf for all its multiples.
    # Since we go in increasing order, the last prime that divides a number
    # is its largest prime factor.
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 2, i):
                lpf[j] = i
                
    best_n = 10000
    best_max_pf = float('inf')
    
    # Search for the n that minimizes the max prime factor of n and n+1
    # We only need to check n from 10000 to 1000000
    # lpf[n] and lpf[n+1] are available
    # We can vectorize the calculation of max(lpf[n], lpf[n+1])
    # range is 10000 to 1000000 inclusive.
    # Indices for n: 10000 to 1000000
    # Indices for n+1: 10001 to 1000001
    
    n_range = np.arange(10000, N + 1)
    # Get lpf values for n and n+1
    pf_n = lpf[n_range]
    pf_np1 = lpf[n_range + 1]
    
    # Calculate the max prime factor for the pair
    max_pfs = np.maximum(pf_n, pf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_pfs)
    best_n = int(n_range[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END
