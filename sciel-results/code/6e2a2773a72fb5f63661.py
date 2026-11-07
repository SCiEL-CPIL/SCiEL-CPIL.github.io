"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes(limit):
    if limit < 2:
        return []
    flag = bytearray([1]) * (limit + 1)
    flag[0] = 0
    flag[1] = 0
    root = math.isqrt(limit)
    for i in range(2, root + 1):
        if flag[i]:
            step = i
            start = i * i
            flag[start::step] = bytearray(len(flag[start::step]))
    out = []
    for i in range(2, limit + 1):
        if flag[i]:
            out.append(i)
    return out


def construct():
    t0 = time.time()
    hard_deadline = t0 + 55.0

    lo_all = min(lo for lo, hi in P['gaps'])
    hi_all = max(hi for lo, hi in P['gaps'])
    total_span = hi_all - lo_all + 1

    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _sieve_primes(max_gap)

    # Global LPF sieve over [lo_all, hi_all]
    lpf = [0] * total_span
    for i in range(lo_all, hi_all + 1):
        lpf[i - lo_all] = i

    for p in small_primes:
        start = ((lo_all + p - 1) // p) * p
        if start > hi_all:
            continue
        for x in range(start, hi_all + 1, p):
            if lpf[x - lo_all] > p:
                lpf[x - lo_all] = p

    result = [None] * len(P['gaps'])

    for idx, (lo, hi) in enumerate(P['gaps']):
        if time.time() > hard_deadline:
            break
        gap = hi - lo
        best_val = -1
        best_m = None
        for m in range(lo + 1, hi):
            lp = lpf[m - lo_all]
            if lp >= gap:
                if lp > best_val:
                    best_val = lp
                    best_m = m
        if best_m is not None:
            result[idx] = int(best_m)

    return result

# EVOLVE-BLOCK-END
