import time
import random
from math import gcd
from functools import lru_cache

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic witnesses for 64-bit integers
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, max_factor=None):
    """Factor n, optionally stopping early if a factor > max_factor is found."""
    if n == 1:
        return []
    factors = []
    # Small primes trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
            if max_factor is not None and p > max_factor:
                return factors
    if n == 1:
        return factors
    if is_prime(n):
        if max_factor is not None and n > max_factor:
            factors.append(n)
            return factors
        factors.append(n)
        return factors
    
    # Pollard Rho for remaining composite
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if max_factor is not None and m > max_factor:
                factors.append(m)
                return factors
            factors.append(m)
        else:
            d = pollard_rho(m)
            if d == m:
                # Shouldn't happen, but handle it
                if max_factor is not None and m > max_factor:
                    factors.append(m)
                    return factors
                factors.append(m)
            else:
                stack.append(d)
                stack.append(m // d)
    return factors

def sieve(n):
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, n + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Sieve primes up to 10^7
    limit = 10**7
    primes = sieve(limit)
    
    # Precompute small primes up to 10^6 for smoothness checking
    small_primes_limit = 10**6
    small_primes = sieve(small_primes_limit)
    
    best_k = 0
    best_p = 2
    
    # Precompute modular inverses and powers for small primes
    # For each small prime q, we'll track q^j mod (p^2 + i) for i in range
    # Actually, let's use a different approach: for each candidate p,
    # check if p^2 + i has any prime factor > p by checking gcd with product of primes > p
    
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Quick check: if p^2 + 1 is prime, skip
        n0 = p2 + 1
        if is_prime(n0):
            continue
        
        # Factor p^2 + 1 with early termination
        facs = factor(n0, max_factor=p)
        if not facs or max(facs) > p:
            continue
        
        k = 1
        # Check further j
        while k < 200:
            if time.time() > deadline:
                break
            
            n = p2 + k + 1
            
            # Quick primality check
            if is_prime(n):
                # If n is prime and n > p, fail
                if n > p:
                    break
                else:
                    k += 1
                    continue
            
            # Factor with early termination
            facs = factor(n, max_factor=p)
            if not facs or max(facs) > p:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
