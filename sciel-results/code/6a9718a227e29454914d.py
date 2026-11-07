"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 256
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest prime power p^k <= N
    # Each prime power is pairwise coprime with all others (unique prime factor)
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    # Verify pairwise coprimality (should always be true by construction)
    # and that all values are in 1..256 and distinct
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to primes only if something went wrong
                return [1] + primes
        if result[i] < 1 or result[i] > N:
            return [1] + primes
    if len(set(result)) != len(result):
        return [1] + primes
    
    return result
# EVOLVE-BLOCK-END
