import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Generate primes up to 2000 (sqrt(1000000+1) ~ 1000, use 2000 to be safe)
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(2000)
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                lpf = p
                while n % p == 0:
                    n //= p
        if n > 1:
            lpf = n
        return lpf
    
    best_n = 10000
    best_val = largest_prime_factor(10000) * largest_prime_factor(10001)
    # Actually we want max(lpf(n), lpf(n+1)), not product
    best_val = max(largest_prime_factor(10000), largest_prime_factor(10001))
    
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        lpf_n = largest_prime_factor(n)
        lpf_np1 = largest_prime_factor(n + 1)
        val = max(lpf_n, lpf_np1)
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
