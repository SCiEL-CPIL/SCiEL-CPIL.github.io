"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Sieve up to max_gap to get all primes less than max_gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    if max_gap < 2:
        return [None] * len(P['gaps'])

    is_p = bytearray(b'\x01') * (max_gap + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, isqrt(max_gap) + 1):
        if is_p[i]:
            for j in range(i * i, max_gap + 1, i):
                is_p[j] = 0
    primes_small = [i for i in range(2, max_gap) if is_p[i]]

    # For each distinct gap size, precompute the list of forbidden primes (p < gap)
    gap_to_primes = {}
    for lo, hi in P['gaps']:
        g = hi - lo
        if g not in gap_to_primes:
            gap_to_primes[g] = tuple(p for p in primes_small if p < g)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        if gap <= 1:
            continue

        check_primes = gap_to_primes[gap]

        if len(check_primes) == 0:
            # No primes < gap, so any integer strictly between works
            result[i] = int(lo + 1)
            continue

        # Segmented sieve over [lo+1, hi-1]
        size = gap - 1  # number of candidates
        if size <= 0:
            continue

        bad = bytearray(size)

        for p in check_primes:
            rem = (lo + 1) % p
            if rem == 0:
                first = lo + 1
            else:
                first = lo + 1 + (p - rem)
            if first >= hi:
                continue
            start_offset = first - (lo + 1)
            for k in range(start_offset, size, p):
                bad[k] = 1

        # Find the first good position
        for offset in range(size):
            if bad[offset] == 0:
                result[i] = int(lo + 1 + offset)
                break

    # Fallback: for any None entries, do a direct LPF check
    for i, (lo, hi) in enumerate(P['gaps']):
        if result[i] is not None:
            continue
        if time.time() > deadline:
            break
        gap = hi - lo
        if gap <= 1:
            continue
        # Try each candidate in [lo+1, hi-1] and check LPF directly
        for m in range(lo + 1, hi):
            # Check if LPF(m) >= gap
            # m is valid iff m is not divisible by any prime < gap
            valid = True
            for p in primes_small:
                if p >= gap:
                    break
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = int(m)
                break

    return result
# EVOLVE-BLOCK-END
