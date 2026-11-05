"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    # Sieve of Eratosthenes to find primes up to sqrt(N)
    # Primes > sqrt(N) cannot have p^2 <= N, so they don't contribute to powerful part
    limit = int(math.isqrt(N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Compute powerful part P(n) for all n in [1, N]
    P = [1] * (N + 1)
    
    for p in primes:
        # Multiply P[m] by p for every multiple m of p^k, k >= 2
        # This gives p^(e-1) for exponent e >= 2
        pk = p * p
        while pk <= N:
            for m in range(pk, N + 1, pk):
                P[m] *= p
            pk *= p
        # Multiply by p once more for every multiple of p^2 to get p^e
        for m in range(p * p, N + 1, p * p):
            P[m] *= p
    
    # Find n in [10000, N-1] that maximizes P[n] * P[n+1]
    best_n = 10000
    best_val = P[10000] * P[10001]
    
    for n in range(10000, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
