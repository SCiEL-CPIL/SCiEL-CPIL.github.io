import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 1000000
    START = 10000
    
    # 1. Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Compute powerful part for each number
    # P[k] stores the powerful part of k
    P = [1] * (N + 1)
    
    for p in primes:
        # Compute powers of p
        power = p * p
        e = 2
        while power <= N:
            # Iterate multiples of power that are not multiples of p*power
            # k = j * power, where j is not divisible by p
            # j ranges from 1 to N // power
            limit = N // power
            for j in range(1, limit + 1):
                if j % p != 0:
                    k = j * power
                    P[k] *= power
            # Next power
            power *= p
            e += 1
    
    # 3. Find n in [10000, N-1] maximizing P[n] * P[n+1]
    best_n = START
    best_val = P[START] * P[START + 1]
    
    for n in range(START, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
