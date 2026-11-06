import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in bases:
        if a % n == 0:
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
        c = random.randrange(1, n)
        f = lambda x: (x * x + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    if n == 1:
        return []
    if is_prime(n):
        return [n]
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return [p] + factorize(n // p)
    d = pollard_rho(n)
    if d == n:
        return factorize(n)
    return factorize(d) + factorize(n // d)

def sieve(limit):
    is_prime_arr = bytearray([1]) * (limit + 1)
    is_prime_arr[0] = 0
    is_prime_arr[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
    return [i for i, val in enumerate(is_prime_arr) if val]

def batch_run_length(p, primes_le_p, k_max, deadline):
    p_sq = p * p
    residuals = [0] + [p_sq + i for i in range(1, k_max + 1)]
    for q in primes_le_p:
        if q > p:
            break
        if time.time() > deadline:
            break
        r = (-p_sq) % q
        if r == 0:
            start_i = q
        else:
            start_i = r
        for i in range(start_i, k_max + 1, q):
            if residuals[i] <= 1:
                continue
            while residuals[i] % q == 0:
                residuals[i] //= q
    k = 0
    for i in range(1, k_max + 1):
        if residuals[i] == 1:
            k = i
        else:
            break
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 110
    limit = 10 * 10**6
    primes = sieve(limit)
    best_k = 0
    best_p = 2
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        if p < 1000:
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                n = p_sq + i
                if is_prime(n):
                    if n > p:
                        break
                    else:
                        k += 1
                else:
                    factors = factorize(n)
                    if max(factors) > p:
                        break
                    else:
                        k += 1
        else:
            primes_le_p = primes[:idx + 1]
            k = batch_run_length(p, primes_le_p, 200, deadline)
        if k > best_k:
            best_k = k
            best_p = p
    return {'p': best_p, 'k': best_k}
