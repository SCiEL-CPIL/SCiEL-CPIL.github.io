"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Sieve of Eratosthenes to find primes up to 256
    n = 256
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # The set of all primes is a valid solution.
    # Size is 54.
    # Any attempt to include composites requires removing their prime factors,
    # leading to a net loss or no gain in cardinality.
    # Thus, the set of all primes is optimal.
    
    return primes
# EVOLVE-BLOCK-END
