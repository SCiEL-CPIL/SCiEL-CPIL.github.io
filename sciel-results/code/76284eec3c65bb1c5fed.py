"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return list of all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def construct():
    start_time = time.time()
    deadline = start_time + 80  # leave 10s safety margin

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    # Find max gap size to determine sieve limit
    max_gap = max(hi - lo for lo, hi in gaps)
    
    # We need primes < max_gap to check divisibility
    # LPF(m) >= gap_size means m is not divisible by any prime < gap_size
    sieve_limit = max_gap  # primes up to max_gap
    small_primes_all = _sieve_primes_up_to(sieve_limit)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break

        gap_size = hi - lo
        # Primes strictly less than gap_size that would make LPF(m) < gap_size
        small_primes = [p for p in small_primes_all if p < gap_size]

        found = None
        for m in range(lo + 1, hi):
            valid = True
            for p in small_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                found = m
                break
        result[i] = found

    # Ensure all entries are Python int or None (not numpy types)
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        elif isinstance(v, int) and not isinstance(v, bool):
            final_result.append(v)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
