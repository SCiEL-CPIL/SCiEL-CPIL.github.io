"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Determine the full range we need to cover
    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])

    # We need LPF for integers from all_lo+1 to all_hi-1
    lo_bound = all_lo + 1
    hi_bound = all_hi - 1
    size = hi_bound - lo_bound + 1

    # Segmented sieve to compute least prime factor for [lo_bound, hi_bound]
    sqrt_hi = int(math.isqrt(hi_bound)) + 1
    is_prime_small = [True] * (sqrt_hi + 1)
    is_prime_small[0] = False
    if sqrt_hi > 1:
        is_prime_small[1] = False
    for i in range(2, int(math.isqrt(sqrt_hi)) + 1):
        if is_prime_small[i]:
            for j in range(i * i, sqrt_hi + 1, i):
                is_prime_small[j] = False
    small_primes = [i for i, v in enumerate(is_prime_small) if v]

    # Initialize LPF array: 0 means "not yet assigned"
    lpf = [0] * size

    for p in small_primes:
        # Find the first multiple of p in [lo_bound, hi_bound] that is >= p*p
        first = ((lo_bound + p - 1) // p) * p
        if first < p * p:
            first = p * p
        if first > hi_bound:
            continue
        for m in range(first, hi_bound + 1, p):
            idx = m - lo_bound
            if lpf[idx] == 0:
                lpf[idx] = p

    # For numbers still with lpf=0, they are prime; LPF is the number itself
    for i in range(size):
        if lpf[i] == 0:
            lpf[i] = lo_bound + i

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        g = hi - lo
        found = None
        for m in range(lo + 1, hi):
            idx = m - lo_bound
            if lpf[idx] >= g:
                found = m
                break
        result[i] = found

    # Ensure all entries are Python int or None
    final = []
    for v in result:
        if v is None:
            final.append(None)
        else:
            final.append(int(v))

    return final
# EVOLVE-BLOCK-END
