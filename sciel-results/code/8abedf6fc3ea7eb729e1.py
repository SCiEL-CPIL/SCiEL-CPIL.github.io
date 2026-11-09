"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # Precompute least prime factor (LPF) table up to max value needed.
    max_val = max(hi for _, hi in gaps)
    lpf = list(range(max_val + 1))
    for i in range(2, isqrt(max_val) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break

        gap_size = hi - lo
        # For gap_size == 2, no integer exists strictly between lo and hi
        if gap_size <= 2:
            result[i] = None
            continue

        # Search from the middle outward to find a witness
        # This tends to find "well-distributed" composites faster
        mid = (lo + hi) // 2
        candidates = []
        left = mid
        right = mid + 1
        while left > lo or right < hi:
            if left > lo:
                candidates.append(left)
                left -= 1
            if right < hi:
                candidates.append(right)
                right += 1

        found = False
        for m in candidates:
            if lpf[m] >= gap_size:
                result[i] = int(m)
                found = True
                break

        if not found:
            # Fallback: linear scan (should rarely be needed)
            for m in range(lo + 1, hi):
                if lpf[m] >= gap_size:
                    result[i] = int(m)
                    found = True
                    break
            if not found:
                result[i] = None

    # Ensure types are correct (Python int, not numpy)
    for i in range(n):
        if result[i] is not None:
            result[i] = int(result[i])

    return result


# EVOLVE-BLOCK-END
