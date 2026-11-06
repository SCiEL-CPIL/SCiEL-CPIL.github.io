import time
import math
import random

_DEADLINE = 140.0
_WINDOW = 200
_SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _mr(n, a):
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False

def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 1000000007:
        return all(_mr(n, a) for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37))
    return all(_mr(n, a) for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 71, 73, 79, 83, 89, 97))

def _primes_below(n):
    if n < 2:
        return []
    if n <= 1000000:
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0:2] = b'\x00\x00'
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start:n + 1:step] = bytearray(len(sieve[start:n + 1:step]))
        return [i for i in range(2, n + 1) if sieve[i]]
    # segmented
    limit = int(math.isqrt(n)) + 1
    small = bytearray(b'\x01') * (limit + 1)
    small[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if small[i]:
            small[i * i:limit + 1:i] = bytearray(len(small[i * i:limit + 1:i]))
    sp = [i for i in range(2, limit + 1) if small[i]]
    out = []
    B = 1 << 16
    for lo in range(2, n + 1, B):
        hi = min(n, lo + B - 1)
        seg = bytearray(b'\x01') * (hi - lo + 1)
        for p in sp:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            if start == p:
                start += p
            seg[start - lo:hi - lo + 1:p] = bytearray(len(seg[start - lo:hi - lo + 1:p]))
        out.extend(lo + i for i in range(hi - lo + 1) if seg[i])
    return out

def _run_len(p, primes):
    base = p * p + 1
    end = base + _WINDOW - 1
    seg = bytearray(b'\x01') * _WINDOW
    for q in primes:
        if q >= p:
            break
        start = ((base + q - 1) // q) * q
        if start > end:
            continue
        seg[start - base:end - base + 1:q] = bytearray(len(seg[start - base:end - base + 1:q]))
    k = 0
    while k < _WINDOW and seg[k]:
        k += 1
    return k

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    best = [0, 2]
    seen = set()

    def consider(p):
        if p in seen or p <= 1:
            return
        seen.add(p)
        if not is_prime(p):
            return
        pl = _primes_below(p)
        k = _run_len(p, pl)
        if k > best[0]:
            best[0] = k
            best[1] = p

    # Primorial-like primes: p = product of a prime set (small primes), then +1/-1.
    # This makes p a product of many small primes -> dense smooth pool.
    max_prod = 10**9
    def gen(base_set, max_extra, depth, prod, acc):
        if time.time() - t0 > _DEADLINE - 5:
            return
        # candidate: prod+1, prod-1
        for cand in (prod + 1, prod - 1):
            if 2 < cand <= max_prod and cand % 2 == 1:
                consider(cand)
        if depth >= max_extra:
            return
        for i in range(len(_SMALL)):
            q = _SMALL[i]
            if q <= (base_set[-1] if base_set else 1):
                continue
            np = prod * q
            if np > max_prod:
                break
            gen(base_set + [q], max_extra, depth + 1, np, acc)

    # start from a few base primes, allow extra small primes
    for base in ([2], [2, 3], [2, 3, 5], [2, 3, 5, 7]):
        prod = 1
        for q in base:
            prod *= q
        gen(base, 10, 0, prod, None)
        if time.time() - t0 > _DEADLINE - 5:
            break

    # Random dense primes: pick a random set of small primes, product, +/-1
    attempts = 0
    while time.time() - t0 < _DEADLINE - 10 and attempts < 200000:
        attempts += 1
        r = rng.random()
        nsmall = int(2 + r * 8)
        chosen = sorted(rng.sample(_SMALL, min(nsmall, len(_SMALL))))
        prod = 1
        for q in chosen:
            prod *= q
            if prod > max_prod:
                break
        for cand in (prod + 1, prod - 1):
            if 2 < cand <= max_prod and cand % 2 == 1:
                consider(cand)

    # Random primes in [1e5, 1e7] as a broader sweep
    while time.time() - t0 < _DEADLINE - 10:
        p = rng.randrange(100000, 10000000, 2) | 1
        consider(p)

    return {'p': best[1], 'k': best[0]}
