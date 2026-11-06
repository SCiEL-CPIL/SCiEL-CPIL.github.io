import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 1000000
    MAX_VAL = N + 1

    # Generate primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]

    # Initialize powerful part array
    P = [1] * (MAX_VAL + 1)

    # Compute powerful parts
    for p in primes:
        power = p * p
        k = 2
        while power <= MAX_VAL:
            next_power = power * p
            max_q = MAX_VAL // power
            # Optimized loop: iterate q, skip multiples of p
            # Using a while loop with step might not be faster than for+if in Python
            # but let's try to reduce attribute lookups
            P_local = P
            pow_local = power
            p_local = p
            for q in range(1, max_q + 1):
                if q % p_local != 0:
                    P_local[q * pow_local] *= pow_local
            
            power = next_power
            k += 1
            if time.time() - start_time > DEADLINE - 5:
                break
        if time.time() - start_time > DEADLINE - 5:
            break

    # Find n maximizing P[n] * P[n+1]
    best_n = 10000
    best_val = P[10000] * P[10001]

    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > DEADLINE - 2:
            break

    return best_n
