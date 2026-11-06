import math
import time
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()

def _tau_perp_from_factorization(primes):
    """primes: list of distinct primes (squarefree n). Return tau_perp(n)."""
    k = len(primes)
    if k == 0:
        return 0
    if k == 1:
        return 1
    n = 1
    for p in primes:
        n *= p
    # divisors via meet-in-the-middle
    half = k // 2
    A = primes[:half]
    B = primes[half:]
    def gen(sub):
        m = len(sub)
        res = [1]
        for p in sub:
            res = [x for x in res] + [x * p for x in res]
        return res
    da = gen(A)
    db = gen(B)
    divs = [a * b for a in da for b in db]
    divs.sort()
    # gcd of adjacent
    cnt = 0
    prev = divs[0]
    for d in divs[1:]:
        if math.gcd(prev, d) == 1:
            cnt += 1
        prev = d
    return cnt

def _tau_perp(n):
    """Fallback: factorize n (squarefree assumed) and compute."""
    if n < 2:
        return 0
    # trial division by small primes to recover primes
    x = n
    primes = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            primes.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        primes.append(x)
    return _tau_perp_from_factorization(primes)

def construct():
    global _START
    deadline = _START + _DEADLINE - 2.0  # safety margin
    # primes up to 47 (first 15 primes). Product of all 15 ~ 6.1e17 <= 1e18.
    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    # full product check
    full = 1
    for p in base:
        full *= p
    if full > 10**18:
        # trim largest until fits
        while full > 10**18:
            full //= base.pop()
        base = base  # recompute
        full = 1
        for p in base:
            full *= p
    # Evaluate full set first (likely best)
    best = base[:]
    best_score = _tau_perp_from_factorization(base)

    # Also try dropping each single prime to see if a different combo helps
    # (score is generally higher with more primes, but verify a few)
    for i in range(len(base)):
        if time.time() > deadline:
            break
        cand = base[:i] + base[i+1:]
        s = _tau_perp_from_factorization(cand)
        if s > best_score:
            best_score = s
            best = cand

    # Try subsets of size len(base)-2 (drop two) only if time permits
    if time.time() < deadline - 30:
        m = len(base)
        for i in range(m):
            for j in range(i+1, m):
                if time.time() > deadline:
                    break
                cand = base[:i] + base[i+1:j] + base[j+1:]
                s = _tau_perp_from_factorization(cand)
                if s > best_score:
                    best_score = s
                    best = cand
            if time.time() > deadline:
                break

    result = 1
    for p in best:
        result *= p
    if result < 2:
        return 2
    return result

# EVOLVE-BLOCK-END
