"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # need powerful_part for n and n+1, so up to 1000001
    
    # Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    if sqrt_limit >= 0:
        is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Initialize powerful part array
    # pp[x] will store the powerful part of x (product of p^e for e>=2)
    pp = [1] * (LIMIT + 1)
    
    # For each prime, process its powers
    for p in primes:
        # p^2, p^3, ... up to LIMIT
        p2 = p * p
        if p2 > LIMIT:
            continue
        
        # For each power p^k where k >= 2, find multiples
        # A number x has p^e in its factorization with e >= 2 if p^2 divides x
        # The full prime power contribution is p^e where e is the exponent of p in x
        
        # Strategy: for each multiple of p^2, determine the exponent of p in that multiple
        # and multiply pp[multiple] by p^e
        
        # More efficient: iterate over powers
        power = p2
        while power <= LIMIT:
            # All multiples of 'power' have at least this power of p
            # But we need the FULL exponent
            # Let's handle it differently: for each multiple of p^2, compute the exact exponent
            
            # Actually, let's precompute for each number the exponent of each prime
            # That's too slow. Instead:
            # For a given prime p, for each multiple m of p^2, we can compute v_p(m)
            # and then pp[m] *= p^v_p(m)
            
            # But we need to be careful: we should only multiply by p^e once for each prime
            # Let's iterate over multiples of p^2 and compute the exponent
            
            # More efficient approach:
            # For prime p, the numbers that have p^2 dividing them are multiples of p^2
            # For each such number m, v_p(m) = k where p^k || m
            # We can find this by checking m / p^2, m / p^3, etc.
            
            # Let's do it per prime: for each multiple of p^2, determine the full exponent
            m = p2
            while m <= LIMIT:
                # Determine v_p(m)
                temp = m
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp >= 2:
                    pp[m] *= (p ** exp)
                m += p2
            
            # Check time
            if time.time() > deadline - 5.0:
                break
            power *= p
        
        if time.time() > deadline - 5.0:
            break
    
    # Now find the n in [N_MIN, N_MAX] that maximizes pp[n] * pp[n+1]
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline - 2.0:
            break
    
    return best_n
# EVOLVE-BLOCK-END
