"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        for m in range(lo + 1, hi):
            # Find least prime factor of m
            if m % 2 == 0:
                lp = 2
            else:
                lp = m  # default: m is prime
                for d in range(3, isqrt(m) + 1, 2):
                    if m % d == 0:
                        lp = d
                        break
            if lp >= gap:
                result[i] = m
                break
    return result
# EVOLVE-BLOCK-END
