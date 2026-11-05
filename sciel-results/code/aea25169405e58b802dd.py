import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
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
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    if n == 1:
        return []
    if is_prime(n):
        return [n]
    d = pollard_rho(n)
    return factorize(d) + factorize(n // d)

def is_p_smooth(n, p):
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for q in small_primes:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
            if is_prime(n):
                return n <= p
            
    # If n is still large, factorize
    # Optimization: if n < p*p, and n is not 1, and n is prime, we handled it.
    # If n is composite, it must have a factor <= sqrt(n) < p.
    # So we can factorize n.
    factors = factorize(n)
    for f in factors:
        if f > p:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_k = 0
    best_p = 2
    
    # Search primes
    # We will search p up to 10^6 first
    # Then extend if time allows
    
    p = 3
    while p < 10**6:
        if is_prime(p):
            k = 0
            while k < 200:
                if time.time() > deadline:
                    break
                n = p * p + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        p += 2
        
    # Extend search if time allows
    # We can search up to 10^7
    if time.time() < deadline - 20:
        p = 10**6 + 1
        if p % 2 == 0:
            p += 1
        while p < 10**7:
            if is_prime(p):
                k = 0
                while k < 200:
                    if time.time() > deadline:
                        break
                    n = p * p + k + 1
                    if is_p_smooth(n, p):
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
            p += 2
            
    return {'p': best_p, 'k': best_k}
