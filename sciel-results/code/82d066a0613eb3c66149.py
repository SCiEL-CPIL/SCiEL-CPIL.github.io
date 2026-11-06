import time
import math
import random

# ---------------------------------------------------------------------------
# Deterministic Miller-Rabin for 64-bit integers
# ---------------------------------------------------------------------------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        cont = False
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                cont = True
                break
        if cont:
            continue
        return False
    return True

# ---------------------------------------------------------------------------
# Pollard-Brent factorization (randomized)
# ---------------------------------------------------------------------------
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = 128
        g = r = q = 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
        if g != n:
            return g

def _factor(n, out):
    if n <= 1:
        return
    if is_prime_mr(n):
        out.append(n)
        return
    d = pollard_brent(n)
    _factor(d, out)
    _factor(n // d, out)

def factor(n):
    out = []
    _factor(n, out)
    return out

# ---------------------------------------------------------------------------
# p-smooth check with early exit (max prime factor <= p)
# ---------------------------------------------------------------------------
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def is_p_smooth(n, p, depth=0):
    # quick divisibility by small primes
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if n % q == 0:
            n //= q
            while n % q == 0:
                n //= q
    if n == 1:
        return True
    if n > p:
        # n is composite with all prime factors > p, or a prime > p
        if is_prime_mr(n):
            return False
        # composite; find a factor and recurse
        if depth < 12:
            d = pollard_brent(n)
            return is_p_smooth(d, p, depth + 1) and is_p_smooth(n // d, p, depth + 1)
        return False
    # n <= p but > 1: it's a prime (or 1) and thus <= p
    return True

def run_length(p):
    k = 0
    v = p * p
    while k < 200:
        if not is_p_smooth(v + 1, p):
            break
        k += 1
    return k

def sieve(n):
    bs = bytearray([1]) * (n + 1)
    bs[0] = bs[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if bs[i]:
            bs[i * i::i] = bytearray(len(bs[i * i::i]))
    return [i for i in range(n + 1) if bs[i]]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    random.seed(12345)
    t0 = time.time()
    deadline = 145.0  # safety margin under the 150s kill

    best_k = 0
    best_p = 3

    # Phase 1: sieve up to a reasonable limit for fast prime iteration
    LIMIT1 = 3_000_000
    primes = sieve(LIMIT1)

    for p in primes:
        if time.time() - t0 > deadline - 5:
            break
        if p * p > 10**18:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: if we still have time, probe larger primes (sampled)
    if time.time() - t0 < deadline - 10:
        # sample candidate primes in a larger range
        lo = LIMIT1
        hi = 20_000_000
        step = 2
        # find a random-ish stride that still covers a lot
        remaining = deadline - 10 - (time.time() - t0)
        # estimate: each prime test + run_length ~ a few ms
        # just iterate odd numbers, test primality, and if prime, run_length
        n = lo + (lo % 2)
        if n % 2 == 0:
            n += 1
        while n < hi:
            if time.time() - t0 > deadline - 5:
                break
            if is_prime_mr(n):
                k = run_length(n)
                if k > best_k:
                    best_k = k
                    best_p = n
            n += step

    return {'p': best_p, 'k': best_k}
