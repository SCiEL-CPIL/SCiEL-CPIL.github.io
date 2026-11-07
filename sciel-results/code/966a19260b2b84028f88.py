"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Precompute primes up to max possible gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # Sieve of Eratosthenes up to max_gap
    if max_gap >= 2:
        sieve = [True] * (max_gap + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_gap**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, max_gap + 1, i):
                    sieve[j] = False
        primes_below_max = [i for i in range(2, max_gap + 1) if sieve[i]]
    else:
        primes_below_max = []

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Primes to check: all primes < gap (if any prime < gap divides m, lp(m) < gap)
        # We need lp(m) >= gap, so m must not be divisible by any prime < gap
        check_primes = [p for p in primes_below_max if p < gap]

        # Search for m in (lo, hi) not divisible by any prime in check_primes
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            has_small_factor = False
            for p in check_primes:
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                result[i] = m
                break

    # Ensure all entries are proper Python ints or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))

    return final_result
# EVOLVE-BLOCK-END
