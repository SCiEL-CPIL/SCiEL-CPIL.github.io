# EVOLVE-BLOCK-START
import math
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def construct():
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    t0 = time.time()
    deadline = t0 + 85.0  # explicit deadline guard with safety margin

    max_hi = max(hi for _, hi in gaps)
    min_lo = min(lo for _, lo in gaps)

    # Sieve of Eratosthenes to get all primes up to max_hi
    # Also build least prime factor array
    limit = max_hi
    lpf = [0] * (limit + 1)
    # Mark composites and record least prime factor
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            if i * i <= limit:
                for j in range(i * i, limit + 1, i):
                    if lpf[j] == 0:
                        lpf[j] = i

    # Precompute primes list for quick reference
    primes_in_range = set()
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            primes_in_range.add(i)

    for idx, (lo, hi) in enumerate(gaps):
        if time.time() > deadline - 2.0:
            break
        gap_size = hi - lo

        # Strategy: first check if any prime exists in (lo, hi)
        # A prime m has LPF = m, and m > lo >= 10007 >> gap_size, so it's always valid
        found = False
        for m in range(lo + 1, hi):
            if lpf[m] == 0:
                result[idx] = int(m)
                found = True
                break

        if found:
            continue

        # No prime in gap. Need composite m with LPF >= gap_size.
        # This means m has no prime factor < gap_size.
        # Check candidates in order, skipping even numbers and multiples of 3
        # (since gap_size >= 2, and for gap_size > 3, multiples of 3 won't work)
        if gap_size <= 2:
            # gap_size=2: need LPF >= 2, i.e., m is not 1. Any m > 1 works except...
            # Actually LPF >= 2 means m is prime (LPF=m>=2) or m=1 (not in range)
            # Wait: for gap_size=2, we need LPF(m) >= 2. Every integer > 1 has LPF >= 2.
            # So any m in (lo, hi) works.
            result[idx] = int(lo + 1)
            continue

        # For gap_size >= 3: m must not be divisible by any prime < gap_size
        # Small primes to check: 2, 3, 5, 7, 11, 13, ...
        # Quick filter: skip even numbers
        for m in range(lo + 1, hi):
            if m % 2 == 0:
                continue
            if m % 3 == 0 and gap_size > 3:
                continue
            if m % 5 == 0 and gap_size > 5:
                continue
            # Full check: verify LPF(m) >= gap_size
            f = lpf[m]
            if f == 0:
                f = m  # m is prime
            if f >= gap_size:
                result[idx] = int(m)
                found = True
                break

    # Enforce exact types
    out = []
    for v in result:
        if v is None:
            out.append(None)
        else:
            out.append(int(v))
    return out


# EVOLVE-BLOCK-END
