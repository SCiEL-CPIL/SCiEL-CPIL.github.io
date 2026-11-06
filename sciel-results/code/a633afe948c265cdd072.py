import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    N = 1000000
    MAX_VAL = N + 1  # We need P(n+1) for n=N, so up to N+1
    
    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Initialize powerful part array
    P = [1] * (MAX_VAL + 1)
    
    # Compute powerful parts using sieve approach
    # For each prime p, find powers p^k with k>=2
    for p in primes:
        power = p * p
        k = 2
        while power <= MAX_VAL:
            next_power = power * p
            # Multiples of 'power' but not 'next_power' have exactly exponent k for p
            # We multiply p^k into their powerful part
            if next_power > MAX_VAL:
                # All multiples of power have exactly exponent k (no higher power exists)
                for m in range(power, MAX_VAL + 1, power):
                    P[m] *= power
            else:
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        P[m] *= power
            power = next_power
            k += 1
            # Check deadline
            if time.time() - start_time > deadline - 5:
                break
        if time.time() - start_time > deadline - 5:
            break
    
    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    best_n = 10000
    best_val = 0
    for n in range(10000, N + 1):
        val = P[n] * P[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > deadline - 2:
            break
            
    return best_n
# EVOLVE-BLOCK-END
