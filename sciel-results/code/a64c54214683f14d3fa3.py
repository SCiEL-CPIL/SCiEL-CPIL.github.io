"""Rewritten search for large pairwise-coprime sumsets of two bounded 3-sets."""
import random
import math
import time

_LIMIT = 128
_DEADLINE = 80.0  # safety margin below the 90s wall-clock kill


def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit + 1:step] = [False] * (((limit - start) // step) + 1)
    return is_prime


_PRIMES_CACHE = None


def _get_primes():
    global _PRIMES_CACHE
    if _PRIMES_CACHE is None:
        _PRIMES_CACHE = [i for i in range(2, _LIMIT + 1) if _sieve(_LIMIT)[i]]
    return _PRIMES_CACHE


def _valid(A, B):
    """Return (distinct_sum_count, distinct_sum_set) for a candidate, or (0, None) if invalid."""
    A = sorted(A)
    B = sorted(B)
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0, None
    for x in A:
        if not (1 <= x <= _LIMIT):
            return 0, None
    for y in B:
        if not (1 <= y <= _LIMIT):
            return 0, None
    sums = {a + b for a in A for b in B}
    # distinct values must be pairwise coprime
    lst = sorted(sums)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) != 1:
                return 0, None
    return len(lst), sums


def construct():
    start = time.time()
    primes = _get_primes()
    rng = random.Random(20240517)

    # ---- random starts ----
    best_A = None
    best_B = None
    best_score = 0

    def consider(A, B):
        nonlocal best_A, best_B, best_score
        s, _ = _valid(A, B)
        if s > best_score:
            best_score = s
            best_A = sorted(A)
            best_B = sorted(B)

    for _ in range(200000):
        if time.time() > _DEADLINE:
            break
        A = rng.sample(primes, 3)
        B = rng.sample(primes, 3)
        consider(A, B)
        if best_score >= 9:
            break

    if best_A is None:
        # fallback: all-prime construction (still valid, at least 1 sum)
        best_A = primes[:3]
        best_B = primes[3:6]

    # ---- local search: swap one element of A or B with another prime ----
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        A = list(best_A)
        B = list(best_B)
        for pos in range(3):
            if time.time() > _DEADLINE:
                break
            for p in primes:
                if p in A:
                    continue
                old = A[pos]
                A[pos] = p
                s, _ = _valid(A, B)
                if s > best_score:
                    best_score = s
                    best_A = sorted(A)
                    improved = True
                else:
                    A[pos] = old
        for pos in range(3):
            if time.time() > _DEADLINE:
                break
            for p in primes:
                if p in B:
                    continue
                old = B[pos]
                B[pos] = p
                s, _ = _valid(A, B)
                if s > best_score:
                    best_score = s
                    best_B = sorted(B)
                    improved = True
                else:
                    B[pos] = old

    # ---- occasional fresh random restarts if time remains ----
    while time.time() < _DEADLINE:
        A = rng.sample(primes, 3)
        B = rng.sample(primes, 3)
        s, _ = _valid(A, B)
        if s > best_score:
            best_score = s
            best_A = sorted(A)
            best_B = sorted(B)
        if best_score >= 9:
            break

    return [best_A, best_B]
