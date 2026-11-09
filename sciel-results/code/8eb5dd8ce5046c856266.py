"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Returns a list of distinct Python integers from 1..256, pairwise coprime,
    maximizing their exact sum.
    """
    N = 256
    
    # Find all primes up to N using Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    result = [1]
    
    # For each prime, find the largest power p^k <= N
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
        
    return result
# EVOLVE-BLOCK-END
