import sys
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = 50.0  # Safety margin before 60s kill
    
    N = 1000000
    MAX_VAL = N + 1
    
    # 1. Sieve of Eratosthenes to find primes up to sqrt(MAX_VAL)
    # Primes p > sqrt(MAX_VAL) cannot have p^2 <= MAX_VAL, so they don't contribute to powerful part
    limit_sqrt = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit_sqrt + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_sqrt)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_sqrt + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # 2. Compute powerful part for each number 1..MAX_VAL
    # powerful_part[i] = product of p^e for all prime factors p of i with exponent e >= 2
    pow_part = [1] * (MAX_VAL + 1)
    
    # For each prime p, iterate through its powers p^2, p^3, ...
    # and multiply pow_part[m] by p^k for all multiples m of p^k
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            # Iterate through multiples of 'power'
            for m in range(power, MAX_VAL + 1, power):
                pow_part[m] *= power
            power *= p
            if time.time() - start_time > limit_time:
                break
        if time.time() - start_time > limit_time:
            break
            
    # 3. Find n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    max_val = 0
    best_n = 10000
    
    # Standard loop is O(10^6) which is fast enough
    # We can use numpy for vectorization if we want, but pure python loop is ~0.1s
    for n in range(10000, N + 1):
        val = pow_part[n] * pow_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
            
        if time.time() - start_time > limit_time:
            break
            
    return best_n
# EVOLVE-BLOCK-END
