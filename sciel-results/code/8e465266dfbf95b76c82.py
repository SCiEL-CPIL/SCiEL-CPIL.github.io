"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 5 second safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1
    
    # Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    if sqrt_limit >= 0:
        is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Initialize powerful part array
    # pp[x] will store the powerful part of x
    pp = [1] * (LIMIT + 1)
    
    # For each prime, process its powers
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        # For each multiple of p2, divide out all factors of p
        for m in range(p2, LIMIT + 1, p2):
            temp = m
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
            pp[m] *= power
    
    # Find the n in [N_MIN, N_MAX-1] that maximizes pp[n] * pp[n+1]
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    # Check deadline
    if time.time() > DEADLINE:
        return best_n
    
    for n in range(N_MIN, N_MAX):
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        
        # Periodic deadline check
        if n % 10000 == 0 and time.time() > DEADLINE:
            break
    
    return best_n
# EVOLVE-BLOCK-END
