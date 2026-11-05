import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 1000000
    # P[i] = powerful part of i
    P = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # Only primes p where p^2 <= N can contribute to powerful parts
    # So p <= sqrt(N) = 1000
    primes = primes[primes <= int(math.isqrt(N))]
    
    # For each prime, multiply p^k into P[m] for appropriate multiples
    for p in primes:
        if time.time() > deadline:
            break
        p = int(p)
        power = p * p  # p^2
        while power <= N:
            next_power = power * p  # p^(k+1)
            # For each multiple m of power, if p^(k+1) does not divide m, multiply P[m] by power
            # We need: m % power == 0 and m % next_power != 0
            # This means m is divisible by p^k but not p^(k+1)
            # i.e., the exact power of p dividing m is exactly k
            # So we want indices m where m = power * t and t % p != 0
            # m ranges over power, 2*power, 3*power, ... up to N
            # Filter: m % next_power != 0
            # Equivalent: (m // power) % p != 0
            
            # Create array of multiples
            multiples = np.arange(power, N + 1, power, dtype=np.int64)
            # Filter those where (m // power) % p != 0
            t_vals = multiples // power
            mask = (t_vals % p) != 0
            valid_multiples = multiples[mask]
            P[valid_multiples] *= power
            power = next_power
            if power > N:
                break
    
    # Find n in [10000, 1000000] that maximizes P[n] * P[n+1]
    # P is 1-indexed, so P[n] is the powerful part of n
    best_val = 0
    best_n = 10000
    for n in range(10000, N):  # n+1 must be <= N
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return int(best_n)
