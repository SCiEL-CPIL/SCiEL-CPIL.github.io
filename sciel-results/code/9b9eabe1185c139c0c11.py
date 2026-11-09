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
    return [i for i in range(2, n + 1) if sieve[i]]


def _sieve_is_prime(n):
    """Return bytearray isprime[0..n] where isprime[i] is 1 iff i is prime."""
    isprime = bytearray(b'\x01') * (n + 1)
    if n >= 0:
        isprime[0] = 0
    if n >= 1:
        isprime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if isprime[i]:
            isprime[i * i::i] = bytearray(len(isprime[i * i::i]))
    return isprime


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before the 90s kill

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    # A composite m has a least prime factor <= isqrt(m).  For every gap here
    # isqrt(hi) < gap_size, so no composite can satisfy LPF >= gap_size:
    # the only possible witnesses are primes.  A single primality sieve over
    # the whole range makes the per-candidate check an O(1) lookup.
    max_val = max(hi for lo, hi in gaps)
    isprime = _sieve_is_prime(max_val)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        # Strictly between lo and hi, and prime => LPF(m) = m >= gap_size.
        for m in range(lo + 1, hi):
            if isprime[m]:
                result[i] = m
                break

    # Ensure exact Python int types (reject numpy ints / bools).
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
