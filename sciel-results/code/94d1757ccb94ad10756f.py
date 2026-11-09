# EVOLVE-BLOCK-START
import math
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _lpf_sieve(limit):
    """Return an array lpf where lpf[n] is the least prime factor of n (0 if n<2 or prime)."""
    lpf = [0] * (limit + 1)
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i * i, limit + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    return lpf


def construct():
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    t0 = time.time()
    deadline = t0 + 60.0  # explicit deadline guard

    max_hi = max(hi for _, hi in gaps)
    # Precompute least prime factor for every number up to the largest prime.
    lpf = _lpf_sieve(max_hi)

    for idx, (lo, hi) in enumerate(gaps):
        if time.time() > deadline - 5.0:
            break
        gap_size = hi - lo
        # A candidate m strictly between lo and hi is valid iff its least prime
        # factor is >= gap_size. For primes (lpf[m]==0) the least factor is m itself.
        for m in range(lo + 1, hi):
            f = lpf[m]
            if f == 0:
                f = m  # m is prime -> least prime factor is m
            if f >= gap_size:
                result[idx] = int(m)
                break

    # Enforce exact types: Python int (never bool/NumPy) or None.
    out = []
    for v in result:
        if v is None:
            out.append(None)
        else:
            out.append(int(v))
    return out


# EVOLVE-BLOCK-END
