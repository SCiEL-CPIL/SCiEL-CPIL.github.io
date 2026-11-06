"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Approach:
    1. We know phi(n) = n * prod_{p|n} (1 - 1/p) = prod_{p^a || n} p^(a-1) * (p-1).
    2. For phi(n) to be divisible by 120 = 2^3 * 3 * 5, the product of (p-1) terms
       (times the remaining prime powers) must supply these factors.
    3. The primes p that can appear in n must have p-1 dividing some multiple related to 120.
       More precisely, for the totient to be divisible by 120, the prime factors of n
       are constrained. However, the simplest robust approach is to use a sieve to compute
       phi(n) for all n up to 1,000,000 and find the smallest n with phi(n) % 120 == 0.
    
    Since N=1,000,000 is small, a linear-time sieve for phi is efficient enough.
    """
    deadline = time.time() + 50  # Safety margin
    
    N = 1000000
    phi = list(range(N + 1))
    
    # Sieve of Eratosthenes-style computation of phi
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
        
        if time.time() > deadline:
            break
    
    # Find the smallest n with phi[n] % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    return -1
# EVOLVE-BLOCK-END
