import math
import time

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 80.0  # 80 seconds, leaving 10s margin for 90s limit
    
    # 1. Sieve of Eratosthenes to find primes up to sqrt(N_MAX)
    sqrt_limit = int(math.isqrt(N_MAX))
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Compute powerful part for all n in [1, N_MAX+1]
    # pp[x] stores the powerful part of x
    pp = [1] * (N_MAX + 2)
    
    for p in primes:
        p2 = p * p
        if p2 > N_MAX:
            break
            
        e = 2
        power = p2
        while power <= N_MAX:
            # Next power to check for divisibility to ensure exact exponent
            next_power = power * p
            
            # Iterate multiples of 'power'
            # We only want to update if 'power' is the EXACT highest power of p dividing m
            # i.e., m is divisible by power but not by next_power
            
            for m in range(power, N_MAX + 1, power):
                # Check if m is NOT divisible by next_power
                # Equivalent to (m // power) % p != 0
                if (m // power) % p != 0:
                    pp[m] *= power
                    
            # Check time
            if time.time() > DEADLINE:
                break
                
            e += 1
            power = next_power
            
        if time.time() > DEADLINE:
            break
            
    if time.time() > DEADLINE:
        # Return a safe default if we ran out of time during precomputation
        # Though unlikely given the complexity
        return N_MIN

    # 3. Find n in [N_MIN, N_MAX] maximizing pp[n] * pp[n+1]
    max_product = 0
    best_n = N_MIN
    
    for n in range(N_MIN, N_MAX + 1):
        prod = pp[n] * pp[n + 1]
        if prod > max_product:
            max_product = prod
            best_n = n
            
        # Optional: check time, but this loop is O(N) ~ 10^6, very fast
        # if time.time() > DEADLINE:
        #     break

    return best_n
