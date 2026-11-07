# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _gen_primes(max_gap)

    # Precompute avoid-primes per gap size
    gap_to_avoid = {}
    for lo, hi in P['gaps']:
        g = hi - lo
        if g not in gap_to_avoid:
            gap_to_avoid[g] = tuple(p for p in small_primes if p < g)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        g = hi - lo
        avoid = gap_to_avoid[g]

        # Search from both ends toward the middle
        found = False
        # Try from left
        for m in range(lo + 1, hi):
            ok = True
            for p in avoid:
                if m % p == 0:
                    ok = False
                    break
            if ok:
                result[i] = m
                found = True
                break

        if not found:
            # Try from right
            for m in range(hi - 1, lo, -1):
                ok = True
                for p in avoid:
                    if m % p == 0:
                        ok = False
                        break
                if ok:
                    result[i] = m
                    break

    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            final_result.append(int(r))
    return final_result
# EVOLVE-BLOCK-END
