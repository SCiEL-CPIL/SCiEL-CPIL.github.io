import math
import random
import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def is_p_smooth(n, p, primes):
    m = n
    # Iterate primes up to min(p, sqrt(m))
    # Optimization: if q*q > m, break
    for q in primes:
        if q > p:
            break
        if q * q > m:
            break
        while m % q == 0:
            m //= q
    if m == 1:
        return True
    return m <= p

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Sieve primes up to 2e6
    # This gives ~150k primes
    # Check speed: 150k * 5ms = 750s. Too slow.
    # We need to be smarter.
    
    # Let's sieve up to 500,000 first. ~40k primes.
    # 40k * 5ms = 200s. Still tight.
    
    # Let's sieve up to 200,000. ~18k primes.
    # 18k * 5ms = 90s. Safe.
    
    limit = 200000
    primes = sieve_primes(limit)
    
    best_p = 7
    best_k = 1
    
    for p in primes:
        if time.time() > deadline:
            break
        if p < 3:
            continue
        
        k = 0
        # Check j=1..200
        # Early break if k > best_k? No, we want max k.
        # But if k reaches 200, break.
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p, primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
