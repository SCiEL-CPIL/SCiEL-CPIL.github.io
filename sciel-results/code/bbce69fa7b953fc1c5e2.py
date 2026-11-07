"""LPF-SIEVE witness search: single global least-prime-factor sieve, then linear scan per gap."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _lpf_sieve(n):
    """Return list lpf where lpf[m] = least prime factor of m (or m if prime)."""
    lpf = list(range(n + 1))
    for i in range(2, int(n**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf


def construct():
    deadline = time.time() + 85.0  # safety margin vs 90s wall-clock

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # Precompute LPF up to the maximum hi in the gaps
    max_val = max(hi for lo, hi in gaps)
    lpf = _lpf_sieve(max_val)

    for idx in range(n):
        if time.time() > deadline:
            break

        lo, hi = gaps[idx]
        gap = hi - lo

        # Find first m in (lo, hi) with least prime factor >= gap
        # If m is prime, lpf[m] = m >= gap (since m > lo >= 10007 and gap <= 20)
        # If m is composite, lpf[m] is its smallest prime factor
        found = None
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                found = m
                break

        if found is not None:
            result[idx] = int(found)

    return result
# EVOLVE-BLOCK-END
