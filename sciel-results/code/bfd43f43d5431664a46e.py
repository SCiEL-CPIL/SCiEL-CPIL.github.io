import numpy as np
import math

def construct():
    # Time limit is 90s. This should take < 1s.
    N_MIN = 10000
    N_MAX = 1000000
    NMAX = N_MAX + 1  # We need LPF up to 1000001
    
    # 1. Sieve of Eratosthenes for primes up to NMAX
    is_prime = np.ones(NMAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(NMAX)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # 2. Compute Largest Prime Factor (LPF)
    # Initialize with 0. Primes will be stamped onto their multiples.
    # Since we iterate primes in ascending order, the last stamp is the largest prime factor.
    lpf = np.zeros(NMAX + 1, dtype=np.int32)
    
    # Get list of primes
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Stamp p onto all multiples of p
        lpf[p::p] = p
        
    # 3. Find n in [N_MIN, N_MAX] minimizing max(LPF(n), LPF(n+1))
    # lpf[n] for n in N_MIN..N_MAX
    # lpf[n+1] for n in N_MIN..N_MAX -> indices N_MIN+1 .. N_MAX+1
    
    # Extract relevant slices
    # Note: lpf array is 0-indexed.
    # n ranges from N_MIN to N_MAX
    # lpf[n] is lpf[N_MIN:N_MAX+1]
    # lpf[n+1] is lpf[N_MIN+1:N_MAX+2]
    
    curr_lpf = lpf[N_MIN:N_MAX+1]
    next_lpf = lpf[N_MIN+1:N_MAX+2]
    
    # Compute score for each n
    max_lpf = np.maximum(curr_lpf, next_lpf)
    
    # Find the minimum score
    min_val = np.min(max_lpf)
    min_idx = np.argmin(max_lpf)
    
    best_n = N_MIN + min_idx
    
    return int(best_n)
