"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Find the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Strategy:
    1. Precompute primes up to sqrt(1000000) = 1000.
    2. Iterate n from 1 to 1000000.
    3. For each n, compute phi(n) by factorizing n using the precomputed primes.
    4. Return the first n where phi(n) % 120 == 0.
    """
    LIMIT = 1000000
    # Precompute primes up to sqrt(LIMIT)
    limit_sqrt = int(math.isqrt(LIMIT))
    is_prime = [True] * (limit_sqrt + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_sqrt)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_sqrt + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def compute_phi(n):
        if n == 1:
            return 1
        original_n = n
        result = n
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    for n in range(1, LIMIT + 1):
        if compute_phi(n) % 120 == 0:
            return n
            
    return LIMIT
# EVOLVE-BLOCK-END
