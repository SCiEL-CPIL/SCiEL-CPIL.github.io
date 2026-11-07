"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import sys

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin before 90s kill

    max_candidate = max(hi for lo, hi in P['gaps'])

    # Precompute smallest prime factor up to max_candidate using sieve
    spf = list(range(max_candidate + 1))
    for i in range(2, isqrt(max_candidate) + 1):
        if spf[i] == i:
            for j in range(i * i, max_candidate + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Build list of all primes up to max_candidate
    primes = [i for i in range(2, max_candidate + 1) if spf[i] == i]

    # For each gap size, precompute the set of forbidden primes (primes < gap)
    # A candidate m is valid iff none of the primes < gap divide m
    # i.e., m is coprime to the product of all primes < gap

    # Precompute for each distinct gap size the list of forbidden primes
    gap_forbidden = {}
    for lo, hi in P['gaps']:
        g = hi - lo
        if g not in gap_forbidden:
            gap_forbidden[g] = [p for p in primes if p < g]

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        forbidden = gap_forbidden[gap]

        # Scan candidates from both ends toward the middle, and also from middle outward
        # Strategy: try all candidates in the interval, checking divisibility by forbidden primes
        # To be efficient, we check candidates in an order that might find a witness quickly

        # Try middle-out first (witnesses near the center are often found quickly)
        mid = (lo + hi) // 2
        found = False

        # Build ordered list: middle out
        candidates = []
        for offset in range(0, (hi - lo) // 2 + 1):
            c1 = mid + offset
            c2 = mid - offset
            if lo < c1 < hi:
                candidates.append(c1)
            if lo < c2 < hi and c2 != c1:
                candidates.append(c2)

        for m in candidates:
            if time.time() > deadline:
                break
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = int(m)
                found = True
                break

        if found:
            continue

        # If middle-out didn't find one, do a linear scan (shouldn't happen if middle-out covers all)
        # But as a safety net:
        if time.time() > deadline:
            break
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = int(m)
                break

    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
