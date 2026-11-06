import math
import time
import random

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _SMALL_PRIMES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
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
    if n % 5 == 0:
        return 5
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                x = 3
                y = 3
                c = 1
    return d

def is_p_smooth(n, p):
    if n <= p:
        return True
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n <= p:
                return True
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    d = pollard_rho(n)
    if d == n:
        return n <= p
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_k = 0
    best_p = 2
    
    random.seed(42)
    
    # Search random primes up to 10^9
    while time.time() < deadline:
        # Generate random odd number
        p_candidate = random.randint(10**6, 10**9)
        if p_candidate % 2 == 0:
            p_candidate += 1
        
        # Check if prime
        if not is_prime(p_candidate):
            continue
            
        p = p_candidate
        k = 0
        
        # Check smoothness
        for i in range(1, 201):
            if time.time() > deadline:
                break
            n_val = p * p + i
            if not is_p_smooth(n_val, p):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
