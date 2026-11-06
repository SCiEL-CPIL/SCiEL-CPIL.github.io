"""Rewritten program using segmented sieve for largest prime factor and linear scan."""
import math

# EVOLVE-BLOCK-START
def construct():
    N = 123457
    # We need P(a) for a in [1, N-1] (since b = N-a, and both must be positive)
    # So we compute largest prime factor for all numbers in [1, N-1]
    
    # Sieve of Eratosthenes to find all primes up to N
    is_prime = [True] * N
    is_prime[0] = False
    if N > 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N, i):
                is_prime[j] = False
    
    # List of primes up to N
    primes = [i for i in range(2, N) if is_prime[i]]
    
    # Compute largest prime factor for each number in [1, N-1]
    # Initialize with 1 (for the number 1, P(1)=1)
    lp = [1] * N  # lp[i] = largest prime factor of i
    
    for p in primes:
        for j in range(p, N, p):
            lp[j] = p  # Since we iterate primes in increasing order, the last assignment is the largest
    
    # Now find the minimum of max(lp[a], lp[N-a]) for a in [1, N-1]
    best_val = float('inf')
    best_a = 1
    
    for a in range(1, N):
        b = N - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    a = best_a
    b = N - a
    return [a, b]
# EVOLVE-BLOCK-END
