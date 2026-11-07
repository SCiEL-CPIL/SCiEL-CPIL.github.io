"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    import time

    deadline = time.time() + 85  # safety margin before 90s limit

    # Precompute least prime factor (LPF) table for the entire range
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])
    size = max_val + 1

    # Sieve-based LPF computation: O(n log log n)
    lpf = list(range(size))
    # lpf[n] = least prime factor of n, or n itself if prime
    for i in range(2, isqrt(max_val) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, size, i):
                if lpf[j] == j:
                    lpf[j] = i

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo

        # Search for m in (lo, hi) whose least prime factor >= gap
        # Using precomputed LPF table: O(1) per candidate
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break

        if time.time() > deadline:
            break

    # Ensure all values are Python int, not numpy types
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
