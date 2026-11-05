import math
import time
import random

def construct():
    deadline = time.time() + 113.0

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    SIEVE_LIMIT = 1000000
    primes = sieve(SIEVE_LIMIT)
    small_primes = [p for p in primes if p <= 10000]
    prime_set = set(primes)

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for sp in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == sp:
                return True
            if n % sp == 0:
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
            y = random.randint(2, n - 1)
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
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        for q in small_primes:
            if q > p:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
        if n == 1:
            return True
        if n <= p:
            return True
        if is_prime_mr(n):
            return n <= p
        while n > 1:
            if n <= p:
                return True
            if is_prime_mr(n):
                return n <= p
            d = pollard_brent(n)
            if d == n:
                continue
            if d > p:
                return False
            n //= d
        return True

    def check_run(p, max_k=200):
        p_sq = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = p_sq + i
            if is_p_smooth(n, p):
                k = i
            else:
                break
        return k

    best_k = 0
    best_p = 2

    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    for p in primes:
        if time.time() > deadline:
            break
        if p < 97:
            continue
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 150:
            break

    random.seed(42)
    checked = set(primes)
    for _ in range(5000):
        if time.time() > deadline:
            break
        c = random.randint(1000000, 10**8)
        if c in checked:
            continue
        checked.add(c)
        if not is_prime_mr(c):
            continue
        k = check_run(c)
        if k > best_k:
            best_k = k
            best_p = c

    return {'p': best_p, 'k': best_k}
