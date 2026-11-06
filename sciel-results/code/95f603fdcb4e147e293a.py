import math
import time
import random

# Precompute small primes for trial division
def sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

_SMALL_PRIMES = sieve(10**5)
_SMALL_PRIME_SET = set(_SMALL_PRIMES)

def is_prime(n):
    if n < 2:
        return False
    if n in _SMALL_PRIME_SET:
        return True
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return False
        if p * p > n:
            break
    # Miller-Rabin
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

def largest_prime_factor(n):
    if n == 1:
        return 1
    factors = []
    for p in _SMALL_PRIMES:
        if n % p == 0:
            while n % p == 0:
                factors.append(p)
                n //= p
            if n == 1:
                return max(factors)
    stack = [n]
    max_f = 1
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
        else:
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
    return max_f

def is_p_smooth(n, p):
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    m = n
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if m % q == 0:
            while m % q == 0:
                m //= q
            if m == 1:
                return True
    if m == 1:
        return True
    if is_prime(m):
        return m <= p
    # m is composite
    # Check if all factors are <= p
    lf = largest_prime_factor(m)
    return lf <= p

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    best_k = 0
    best_p = 2

    # Sieve primes up to 10^7
    limit = 10**7
    primes = sieve(limit)

    for p in primes:
        if time.time() > deadline:
            break
        
        k = 0
        for i in range(1, 201):
            n_val = p * p + i
            if is_p_smooth(n_val, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
