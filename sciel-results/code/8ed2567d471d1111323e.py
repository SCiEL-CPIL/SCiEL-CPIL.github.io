import math
import time

def construct():
    N = 1000000
    start_time = time.time()
    deadline = 85.0 # 85s limit to be safe within 90s
    
    # 1. Find primes up to sqrt(N)
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # 2. Sieve for powerful parts
    pp = [1] * (N + 1)
    
    for p in primes:
        power = p * p
        e = 2
        while power <= N:
            next_power = power * p
            # Iterate through multiples of power
            # If m is a multiple of power but not next_power, exponent is exactly e
            for m in range(power, N + 1, power):
                if m % next_power != 0:
                    pp[m] *= power
            power = next_power
            e += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # 3. Find max product
    max_score = 0
    best_n = 10000
    
    # n must be in [10000, 999999] so that n+1 <= 1000000
    for n in range(10000, N):
        score = pp[n] * pp[n+1]
        if score > max_score:
            max_score = score
            best_n = n
        if time.time() > deadline:
            break
            
    return best_n
