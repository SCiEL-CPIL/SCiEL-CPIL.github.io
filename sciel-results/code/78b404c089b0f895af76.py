import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def sieve(limit):
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i, prime in enumerate(is_prime) if prime]
    
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve(SMALL_PRIME_LIMIT)
    
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def is_p_smooth(p, n):
        """Check if all prime factors of n are <= p"""
        temp = n
        max_trial = min(p, SMALL_PRIME_LIMIT)
        for q in small_primes:
            if q > max_trial:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        return temp == 1 or temp <= p
    
    def check_run(p):
        p_sq = p * p
        k = 0
        for j in range(1, 201):
            if not is_p_smooth(p, p_sq + j):
                break
            k = j
        return k
    
    # Generate primes up to 3*10^6
    PRIME_LIMIT = 3 * 10**6
    primes_p = sieve(PRIME_LIMIT)
    
    best_k = 0
    best_p = 2
    
    for p in primes_p:
        if time.time() > deadline:
            break
        
        k = check_run(p)
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
    
    # Try random large primes if time permits
    if best_k < 10 and time.time() < deadline - 10:
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            while True:
                candidate = random.randint(10**6, 10**9)
                if candidate % 2 == 0:
                    candidate += 1
                if is_prime_mr(candidate):
                    break
            
            k = check_run(candidate)
            if k > best_k:
                best_k = k
                best_p = candidate
                if best_k >= 200:
                    break
    
    return {'p': best_p, 'k': best_k}
