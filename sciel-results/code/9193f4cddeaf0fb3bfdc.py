"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 512
    # Sieve of Eratosthenes to find primes < N
    is_prime = [True] * N
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N) if is_prime[i]]
    return primes
# EVOLVE-BLOCK-END
