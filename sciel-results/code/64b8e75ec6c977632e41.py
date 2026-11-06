import math
import time
import random

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
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r += 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g

def factorize(n, factors):
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    factorize(d, factors)
    factorize(n // d, factors)

def construct():
    start_time = time.time()
    deadline = 110
    
    # Small primes for trial division
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2
    
    def is_p_smooth(n, p):
        if n == 1:
            return True
        if is_prime(n):
            return False
        
        for q in small_primes:
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if is_prime(n):
                    return n <= p
                break
        
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
            
        factors = []
        factorize(n, factors)
        for f in factors:
            if f > p:
                return False
        return True

    def check_run(p):
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if not is_p_smooth(n, p):
                break
            k += 1
        return k

    # Scan primes from 10^9 downwards
    p = 10**9
    if p % 2 == 0:
        p -= 1
        
    while p > 10**6:
        if time.time() > deadline:
            break
            
        if is_prime(p):
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        p -= 2

    return {'p': best_p, 'k': best_k}
