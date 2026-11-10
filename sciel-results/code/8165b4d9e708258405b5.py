import numpy as np

def construct():
    N = 1000000
    
    # We want to find n in [10000, N] that minimizes max(lpf(n), lpf(n+1))
    # lpf(x) = largest prime factor of x
    #
    # Key insight: we can compute lpf for all numbers up to N+1 using a sieve.
    # Then we just need to find the n that minimizes max(lpf[n], lpf[n+1]).
    #
    # Optimization: Instead of computing lpf for all numbers, we can think about
    # what makes max(lpf[n], lpf[n+1]) small.
    # 
    # For max(lpf[n], lpf[n+1]) to be <= K, both n and n+1 must be K-smooth
    # (all prime factors <= K).
    #
    # Strategy: 
    # 1. Compute lpf for all numbers up to N+1 using a linear-ish sieve
    # 2. Find the minimum of max(lpf[n], lpf[n+1]) for n in [10000, N]
    #
    # The current approach already does this. Let me think if there's a better way.
    #
    # Actually, the key optimization is in the sieve itself. The current sieve
    # iterates over all primes and updates multiples. This is O(N log log N) which
    # should be fine for N=1000000.
    #
    # Let me try a different approach: use a modified Sieve of Eratosthenes 
    # that tracks the largest prime factor more efficiently.
    
    # lpf[i] = largest prime factor of i
    lpf = np.zeros(N + 2, dtype=np.int32)
    
    # For each prime p, mark all multiples with p as their largest prime factor
    # Since we process primes in increasing order, the last prime that marks a number
    # is its largest prime factor.
    
    # First, let's find all primes up to N+1 using a simple sieve
    is_prime = np.ones(N + 2, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(N + 1)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Now, for each prime, update lpf for its multiples
    # Process primes in increasing order so that larger primes overwrite smaller ones
    primes = np.where(is_prime)[0]
    
    for p in primes:
        if p > N + 1:
            break
        # Update lpf for all multiples of p
        for j in range(p, N + 2, p):
            lpf[j] = p
    
    # Now find n in [10000, N] that minimizes max(lpf[n], lpf[n+1])
    # Use vectorized operations
    
    # We need lpf[n] for n in [10000, N] and lpf[n+1] for n in [10000, N]
    # So we need lpf[10000] to lpf[N+1]
    
    pf_n = lpf[10000:N+1]       # lpf[10000], lpf[10001], ..., lpf[N]
    pf_np1 = lpf[10001:N+2]     # lpf[10001], lpf[10002], ..., lpf[N+1]
    
    max_pfs = np.maximum(pf_n, pf_np1)
    
    min_idx = np.argmin(max_pfs)
    best_n = 10000 + min_idx
    
    return int(best_n)
