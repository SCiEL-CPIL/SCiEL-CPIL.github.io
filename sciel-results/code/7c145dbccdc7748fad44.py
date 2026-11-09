import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1
    
    # Efficient computation of Powerful Part P(x)
    # P(x) = product of p^e where e >= 2 in factorization of x.
    
    P = [1] * limit
    
    # Sieve of Eratosthenes for primes
    is_prime = [True] * limit
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    
    for p in primes:
        p2 = p * p
        if p2 >= limit:
            break
        # For each multiple of p2, determine the exact power of p
        # We iterate through powers p^k
        power = p2
        while power < limit:
            # Multiples of 'power' that are NOT multiples of 'power*p'
            # have exactly exponent k for prime p.
            next_power = power * p
            for m in range(power, limit, power):
                if m % next_power != 0:
                    P[m] *= power
            power = next_power
            
    # Find max P[n] * P[n+1]
    best_n = N_MIN
    best_val = P[N_MIN] * P[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() > DEADLINE:
            break
            
    return best_n
