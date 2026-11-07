"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])

    # Build least-prime-factor table up to all_hi
    size = all_hi + 1
    lpf = list(range(size))
    limit = isqrt(all_hi)
    for i in range(2, limit + 1):
        if lpf[i] == i:
            step = i
            start = i * i
            for j in range(start, size, step):
                if lpf[j] == j:
                    lpf[j] = i

    result = [None] * len(P['gaps'])

    for idx in range(len(P['gaps'])):
        if time.time() > deadline:
            break
        lo, hi = P['gaps'][idx]
        gap = hi - lo
        found = False
        # Search from the middle outward to find a witness faster
        mid = (lo + hi) // 2
        # Try mid first, then alternate left/right
        offset = 0
        while not found and offset <= mid - lo:
            for m in [mid + offset, mid - offset]:
                if lo < m < hi and lpf[m] >= gap:
                    result[idx] = int(m)
                    found = True
                    break
            if not found:
                offset += 1
        # Fallback: linear scan if the symmetric search didn't find one
        if not found:
            for m in range(lo + 1, hi):
                if lpf[m] >= gap:
                    result[idx] = int(m)
                    found = True
                    break

    return result
# EVOLVE-BLOCK-END
