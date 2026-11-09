"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1
    
    # Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = np.ones(sqrt_limit + 1, dtype=bool)
    is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.array([i for i, prime in enumerate(is_prime) if prime])
    
    # Initialize powerful part array using int64
    pp = np.ones(LIMIT + 1, dtype=np.int64)
    
    # For each prime, process its powers
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        
        # For each multiple of p2, compute the full prime power factor
        # We need to find the largest p^e that divides m where e >= 2
        # Start with p^2, then multiply by p while divisible
        
        # Create array of multiples
        multiples = np.arange(p2, LIMIT + 1, p2)
        
        # For each multiple, compute the full power of p dividing it
        # We can do this iteratively: start with p^2, then check if divisible by p again
        current_power = p2
        while current_power * p <= LIMIT:
            # For multiples that are divisible by current_power * p, extend the power
            next_power = current_power * p
            # Multiples of next_power get their power extended
            mult_next = np.arange(next_power, LIMIT + 1, next_power)
            # But we need to be careful: we only want to extend if the number is actually divisible by the higher power
            # A number m is divisible by p^k iff m is a multiple of p^k
            # So for multiples of p^(e+1), we add one more factor of p
            # This means: for each m that is a multiple of p^(e+1), pp[m] *= p
            pp[mult_next] *= p
            current_power = next_power
        
        if time.time() > DEADLINE:
            break
    
    # Find the n in [N_MIN, N_MAX] that maximizes pp[n] * pp[n+1]
    # Use NumPy for the scan
    products = pp[N_MIN:N_MAX+1] * pp[N_MIN+1:N_MAX+2]
    best_idx = np.argmax(products)
    best_n = N_MIN + best_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
