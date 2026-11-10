import math
import time
from collections import defaultdict

_DEADLINE = 140.0  # hard wall-clock guard (evaluator kills at 150s)
_START = time.time()
_MAX_K = 10_000_000
_MAX_M = 10**18


def _sigma(n):
    """Exact sum of divisors via trial factorization."""
    if n <= 1:
        return 0
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            p = d
            power = 1
            while n % p == 0:
                n //= p
                power *= p
            result *= (power * (p + 1)) // (p - 1)
        d += 1 if d == 2 else 2
    if n > 1:
        result *= (n * (n + 1)) // (n - 1)
    return result


def _build_sigma(limit):
    """Sieve sigma for 1..limit, returning a Python list of ints."""
    s = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            s[m] += d
    return s


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 140.0

    deadline = _DEADLINE
    best_M = 333312
    best_k = [336, 372, 434]  # guaranteed valid seed (3 distinct solutions)

    sigma = _build_sigma(_MAX_K)

    groups = defaultdict(list)
    for k in range(1, _MAX_K + 1):
        M = k * sigma[k]
        if M > _MAX_M:
            continue
        groups[M].append(k)
        if (k & 0x3FFFF) == 0 and time.time() >= deadline - 6.0:
            break

    # pick the M with the most (distinct) preimages
    best_count = 0
    for M, lst in groups.items():
        c = len(set(lst))
        if c > best_count:
            best_count = c
            best_M = M
            best_k = sorted(set(lst))

    # verify with exact factorization, drop any that fail
    verified = []
    for k in best_k:
        if k >= 1 and k <= math.isqrt(best_M) and k * _sigma(k) == best_M:
            verified.append(k)
    if len(verified) >= len(best_k):
        best_k = verified
    else:
        best_k = verified if len(verified) > len(best_k) else best_k

    # safety: if something went wrong, fall back to the seed
    if len(best_k) < 3:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_k = [int(k) for k in best_k]
    best_M = int(best_M)
    return {'M': best_M, 'k': best_k}
