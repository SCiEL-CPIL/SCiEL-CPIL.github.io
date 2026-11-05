"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Compute powerful part P(n) for all n in [1, N]
    # P(n) = product of p^e for all prime powers p^e dividing n with e >= 2
    # i.e., the largest powerful divisor of n
    # Use multiplicative sieve: for each prime p, for each k >= 2,
    # multiply p^k into P[m] for every multiple m of p^k.
    P = [1] * (N + 1)
    
    for p in primes:
        pk = p * p
        while pk <= N:
            for m in range(pk, N + 1, pk):
                P[m] *= pk
            pk *= p
    
    # Now find n in [10000, 1000000] that maximizes P[n] * P[n+1]
    best_n = 10000
    best_val = P[10000] * P[10001]
    
    for n in range(10000, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
