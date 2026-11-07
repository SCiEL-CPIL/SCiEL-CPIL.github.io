"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    min_lo = min(lo for lo, hi in P['gaps'])
    max_hi = max(hi for lo, hi in P['gaps'])
    max_candidate = max_hi

    # Build SPF table using NumPy for speed
    spf = np.arange(max_candidate + 1, dtype=np.int32)
    for i in range(2, isqrt(max_candidate) + 1):
        if spf[i] == i:
            j_start = i * i
            spf[j_start::i] = np.minimum(spf[j_start::i], i)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Vectorized check: find first m in (lo, hi) with spf[m] >= gap
        candidates = np.arange(lo + 1, hi)
        if len(candidates) > 0:
            mask = spf[candidates] >= gap
            if mask.any():
                idx = np.argmax(mask)  # first True
                result[i] = int(candidates[idx])

    return result
# EVOLVE-BLOCK-END
