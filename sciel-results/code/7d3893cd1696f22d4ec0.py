# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _lpf_sieve(n):
    """Least prime factor array up to n (1-indexed). lpf[x] = smallest prime dividing x, or x if prime."""
    lpf = list(range(n + 1))
    if n >= 1:
        lpf[1] = 1
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf


def construct():
    deadline = time.time() + 50.0  # safety margin under 60s
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_val = max(hi for lo, hi in gaps)
    lpf = _lpf_sieve(max_val)

    # Pre-compute primes less than max possible gap for quick filtering
    max_gap = max(hi - lo for lo, hi in gaps)
    small_primes = [p for p in range(2, max_gap) if lpf[p] == p]

    # For each pair, find a candidate m in (lo, hi) with lpf[m] >= gap
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap = hi - lo
        found = False
        # Try all candidates in the gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = int(m)
                found = True
                break
        if not found:
            # Secondary check: manually verify lpf for each candidate
            # (in case of any sieve issue) - this is a safety net
            for m in range(lo + 1, hi):
                # Compute lpf manually
                temp = m
                min_p = m
                for p in small_primes:
                    if p * p > temp:
                        break
                    if temp % p == 0:
                        min_p = p
                        break
                else:
                    # No small prime factor, check if m itself is prime
                    # If m is prime, lpf[m] = m
                    # But m is between consecutive primes, so m is composite
                    # min_p remains m if no divisor found, but m is composite so this shouldn't happen
                    # Actually, if no small prime divides m, then lpf[m] >= max_gap > gap, so valid
                    if min_p >= gap:
                        result[i] = int(m)
                        found = True
                        break
    return result
# EVOLVE-BLOCK-END
