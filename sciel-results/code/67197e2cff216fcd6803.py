# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    t0 = time.monotonic()
    D = t0 + 55

    hi_max = max(p[1] for p in P['gaps'])

    # Sieve of Eratosthenes up to hi_max - 1 (max possible interior value)
    N = hi_max - 1
    is_comp = bytearray(N + 1)
    is_comp[0:2] = b'\x01\x01'
    for r in range(2, isqrt(N) + 1):
        if not is_comp[r]:
            is_comp[r * r :: r] = b'\x01' * len(range(r * r, N + 1, r))

    # Precompute primes up to sqrt(N) for trial division
    sqrt_N = isqrt(N)
    primes_small = []
    for q in range(2, sqrt_N + 1):
        if not is_comp[q]:
            primes_small.append(q)

    def lpf(m):
        # m is composite (1 < m < N, not prime)
        for p in primes_small:
            if p * p > m:
                break
            if m % p == 0:
                return p
        return m

    res = []
    for lo, hi in P['gaps']:
        if time.monotonic() > D:
            res.append(None)
            continue
        g = hi - lo
        found = None
        for m in range(lo + 1, hi):
            if not is_comp[m]:
                # m is prime => lpf(m) = m >= g always holds for interior m
                found = m
                break
            else:
                if lpf(m) >= g:
                    found = m
                    break
        res.append(found)

    return res
# EVOLVE-BLOCK-END
