"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_lpf(n):
    """Return array lpf where lpf[i] is the least prime factor of i (or i if prime)."""
    lpf = list(range(n + 1))
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin out of 90s limit

    # Determine the maximum value we need to sieve up to
    max_val = max(hi for lo, hi in P['gaps'])

    # Sieve for least prime factor (LPF) up to max_val
    # lpf[i] = least prime factor of i, or i itself if i is prime
    lpf = _sieve_lpf(max_val)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need an integer m strictly between lo and hi
        # such that lpf[m] >= gap
        # Since m is in (lo, hi), and lo >= 10007, m is always > gap (gap <= 28)
        # So if m is prime, lpf[m] = m >= gap is automatically satisfied.
        # We just scan for the first m where lpf[m] >= gap.
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
        # If not found, result[i] remains None

    # Ensure types are correct (Python int)
    for i in range(len(result)):
        if result[i] is not None:
            result[i] = int(result[i])

    return result
# EVOLVE-BLOCK-END
