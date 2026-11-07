"""Rewritten: search divisor-sum fibers for ratio near 3/2."""
import time
from fractions import Fraction

MAXV = 10_000_000
DEADLINE = 75.0
_t0 = time.time()

def _time_ok():
    return (time.time() - _t0) < DEADLINE

def _build_sigma():
    N = MAXV
    sigma = [1] * (N + 1)
    for p in range(2, N + 1):
        if sigma[p] == 1:  # prime
            for m in range(p, N + 1, p):
                t = m
                pe = 1
                while t % p == 0:
                    t //= p
                    pe *= p
                factor = (pe * p - 1) // (p - 1)
                sigma[m] *= factor
    return sigma

_sigma = None

def construct():
    global _sigma
    if _sigma is None:
        _sigma = _build_sigma()
    sigma = _sigma

    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, MAXV + 1):
        groups[sigma[n]].append(n)
        if not _time_ok():
            break

    best = None  # (err, a, b)
    for s, lst in groups.items():
        if len(lst) < 2:
            continue
        lst.sort()
        for i, n in enumerate(lst):
            ideal = 1.5 * n
            lo, hi = 0, len(lst) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lst[mid] < ideal:
                    lo = mid + 1
                else:
                    hi = mid
            for j in (lo, lo - 1):
                if 0 <= j < len(lst):
                    m = lst[j]
                    if m == n:
                        continue
                    for a, b in ((n, m), (m, n)):
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue
                        den = 2 * b
                        err = num / den
                        if best is None or err < best[0]:
                            best = (err, a, b)
        if not _time_ok():
            break

    if best is None:
        for s, lst in groups.items():
            if len(lst) >= 2:
                return [int(lst[0]), int(lst[1])]
        return [11, 6]

    return [int(best[1]), int(best[2])]
