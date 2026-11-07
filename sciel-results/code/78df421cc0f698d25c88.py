"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_lpf(n):
    """Return NumPy array lpf where lpf[i] is the least prime factor of i (or i if prime)."""
    lpf = np.arange(n + 1, dtype=np.int32)
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:
            start = i * i
            lpf[start::i] = np.where(lpf[start::i] > i, i, lpf[start::i])
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # 78s safety margin out of 90s limit

    max_val = max(hi for lo, hi in P['gaps'])
    lpf = _sieve_lpf(max_val)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Search for a valid witness m in (lo, hi)
        # A witness must have least prime factor >= gap
        # This means m is not divisible by any prime < gap
        # Strategy: try multiples of primes >= gap, and also check numbers directly
        # Since gap is at most ~16, the search space is small
        # Use the LPF array to quickly check
        lo_idx = lo + 1
        hi_idx = hi - 1
        if lo_idx > hi_idx:
            continue
        
        # Get the slice of lpf values
        candidates = lpf[lo_idx:hi_idx + 1]
        # We need lpf[m] >= gap
        # Find all indices where this is true
        valid_indices = np.where(candidates >= gap)[0]
        
        if len(valid_indices) > 0:
            # Pick the first valid one (or we could be smarter)
            m = int(lo_idx + valid_indices[0])
            # Verify
            if isinstance(m, int) and not isinstance(m, bool) and lo < m < hi and lpf[m] >= gap:
                result[i] = m

    # Ensure types are correct (Python int)
    for i in range(len(result)):
        if result[i] is not None:
            result[i] = int(result[i])

    return result
# EVOLVE-BLOCK-END
