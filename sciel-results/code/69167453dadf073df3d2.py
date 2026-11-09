"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1
    
    # Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    # Primes needed: p such that p^2 <= LIMIT => p <= 1000
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
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
        p_power = p * p
        # p_power is p^k, starting with k=2
        while p_power <= LIMIT:
            p_next_power = p_power * p
            # We want to find all x such that v_p(x) == k (where p_power = p^k)
            # These are multiples of p_power but NOT multiples of p_next_power
            # Iterate through multiples of p_power
            # To optimize, we can iterate j such that x = j * p_power
            # x must be in [1, LIMIT]
            # Condition: x % p_next_power != 0
            
            # Optimization: 
            # Multiples of p_power are p_power, 2*p_power, ...
            # We skip those divisible by p_next_power.
            # Note that p_next_power = p * p_power.
            # So we skip every p-th multiple.
            
            # Let's just iterate all multiples and check, or use step logic.
            # Step logic:
            # Multiples are p_power * 1, p_power * 2, ...
            # If j is divisible by p, then x is divisible by p_next_power.
            # So we only want j NOT divisible by p.
            
            for j in range(1, LIMIT // p_power + 1):
                if j % p == 0:
                    continue
                x = j * p_power
                if x <= LIMIT:
                    pp[x] *= p_power
                else:
                    break
            
            p_power = p_next_power
            
    # Find n in [N_MIN, N_MAX] that maximizes pp[n] * pp[n+1]
    best_n = N_MIN
    max_product = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        prod = pp[n] * pp[n + 1]
        if prod > max_product:
            max_product = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
