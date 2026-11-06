"""Rewritten program: fast p-smooth run search with deadline guard."""
import math
import random
import time

# ---------- deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a % n == 0:
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

# ---------- Pollard-Brent ----------
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
        x = ys = 0
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
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def factor(n, out):
    if n <= 1:
        return
    if is_prime_mr(n):
        out.append(n)
        return
    d = pollard_brent(n)
    factor(d, out)
    factor(n // d, out)

# ---------- sieve ----------
def sieve(limit):
    if limit < 2:
        return []
    bs = bytearray([1]) * (limit + 1)
    bs[0] = bs[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if bs[i]:
            step = i
            start = i * i
            bs[start::step] = bytearray(len(bs[start::step]))
    return [i for i in range(2, limit + 1) if bs[i]]

# ---------- p-smooth check with early exit ----------
def is_p_smooth(n, p, primes):
    if n <= 1:
        return True
    for q in primes:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    if n == 1:
        return True
    if is_prime_mr(n):
        return n <= p
    f = []
    factor(n, f)
    for x in f:
        if x > p:
            return False
    return True

# ---------- scan a run for a given prime p ----------
def scan(p, primes, deadline):
    k = 0
    base = p * p
    while k < 200:
        if time.time() >= deadline:
            break
        if not is_p_smooth(base + k + 1, p, primes):
            break
        k += 1
    return k

# ---------- main ----------
def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    random.seed(12345)

    primes = sieve(100000)
    # extend primes list up to a larger bound for trial division
    # (we only need primes up to p, but p can be large; use small primes for
    #  trial division and Pollard-Brent for the rest)

    best = (0, 2)  # (k, p); p=2 is a valid prime, k>=1

    # Try small primes first (cheap), then larger ones
    # Generate candidate primes up to some limit
    max_p = 2000000  # search primes up to 2e6
    cand = [p for p in primes if p <= max_p]
    if cand and cand[-1] < max_p:
        # extend with a few more if we have time
        pass

    for p in cand:
        if time.time() >= deadline - 2.0:
            break
        if p < 2:
            continue
        k = scan(p, primes, deadline)
        if k > best[0]:
            best = (k, p)

    # If we still have time, try some larger primes via a sparse scan
    # (large p makes p-smooth easier for the same interval length)
    if time.time() < deadline - 5.0:
        # try a few large primes
        large_candidates = [1000003, 1000033, 1000037, 1000039, 1000081,
                            1000099, 1000163, 1000183, 1000189, 1000193,
                            1000199, 1000201, 1000213, 1000249, 1000253,
                            1000291, 1000297, 1000363, 1000367, 1000393]
        for p in large_candidates:
            if time.time() >= deadline - 3.0:
                break
            if p > 10**9:
                continue
            if not is_prime_mr(p):
                continue
            k = scan(p, primes, deadline)
            if k > best[0]:
                best = (k, p)

    return {'p': best[1], 'k': best[0]}
