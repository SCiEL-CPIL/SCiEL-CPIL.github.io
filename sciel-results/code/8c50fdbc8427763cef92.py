"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_lpf(n):
    """Return an array lpf where lpf[m] is the least prime factor of m for m in 0..n."""
    lpf = list(range(n + 1))
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf


def _verify_witness(m, gap):
    """Verify that m's least prime factor is >= gap."""
    if m <= 1:
        return False
    # Check all integers from 2 up to min(sqrt(m), gap-1)
    limit = min(isqrt(m), gap - 1)
    for p in range(2, limit + 1):
        if m % p == 0:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    max_val = max(hi for lo, hi in P['gaps'])
    lpf = _sieve_lpf(max_val)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Find any m in (lo, hi) such that lpf[m] >= gap
        found = False
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                # Double-check with direct verification
                if _verify_witness(m, gap):
                    result[i] = m
                    found = True
                    break

        if time.time() > deadline:
            break

    # Ensure all entries are Python int or None (no bool, no numpy types)
    final = []
    for x in result:
        if x is None:
            final.append(None)
        else:
            final.append(int(x))
    return final
# EVOLVE-BLOCK-END
