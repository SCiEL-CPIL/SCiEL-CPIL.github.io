"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
_start = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def construct():
    # Deadline guard
    if time.time() - _start > _TIMEOUT:
        return [1]
    
    n = 256
    # Sieve to find all primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    
    # For each prime, find the largest power p^k <= 256
    # This is optimal: prime powers of distinct primes are pairwise coprime,
    # so we independently maximize each prime's contribution.
    result = [1]  # include 1
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        result.append(val)
    
    # Ensure types are Python int
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
