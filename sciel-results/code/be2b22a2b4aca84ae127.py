"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin from 60s limit

    # Precompute LPF for all integers in [10008, 10288]
    # Max value in gaps is 10289, so we need up to 10288
    # Min value in gaps is 10007, so we need from 10008
    range_start = 10008
    range_end = 10288  # inclusive
    range_size = range_end - range_start + 1

    # Initialize LPF array: lpf[k] = least prime factor of (range_start + k)
    # Start with 0 meaning "not yet determined"
    lpf = [0] * range_size

    # Sieve for primes up to sqrt(10288) ~ 101.4, so up to 101
    limit = 102
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [i for i, prime in enumerate(is_prime) if prime]

    # For each small prime, mark its multiples in the range
    for p in small_primes:
        # Find the first multiple of p in [range_start, range_end]
        first = ((range_start + p - 1) // p) * p
        for val in range(first, range_end + 1, p):
            idx = val - range_start
            if lpf[idx] == 0:  # Only set if not already marked by a smaller prime
                lpf[idx] = p

    # For remaining entries where lpf is still 0, the number itself is prime
    # (or 1, but our range starts at 10008 so all are > 1)
    for idx in range(range_size):
        if lpf[idx] == 0:
            val = range_start + idx
            lpf[idx] = val  # The number is prime, so LPF is itself

    # Now search each gap
    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break

        gap = hi - lo

        # For gap == 2, m = lo+1 is even, LPF=2, and 2 >= 2 holds
        if gap == 2:
            result[i] = lo + 1
            continue

        # For gap > 2, search odd m in (lo, hi)
        # m must be strictly between lo and hi
        # lo is odd prime, hi is odd prime
        # Odd numbers in (lo, hi): lo+2, lo+4, ..., hi-2
        for m in range(lo + 2, hi, 2):
            idx = m - range_start
            if idx < 0 or idx >= range_size:
                continue
            if lpf[idx] >= gap:
                result[i] = m
                break

    return result
# EVOLVE-BLOCK-END
