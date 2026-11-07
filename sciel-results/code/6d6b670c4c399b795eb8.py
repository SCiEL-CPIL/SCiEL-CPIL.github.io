"""Structural rewrite: precompute a boolean 'valid' array for each gap size.

For a given gap g, an integer m is a valid witness iff LPF(m) >= g, which is
equivalently: m is not divisible by any prime p < g.  We precompute, for every
distinct gap size that appears in the pinned pairs, a NumPy boolean array
valid_g where valid_g[m] is True exactly when no prime < g divides m.  Then for
each (lo, hi) with gap g we just look for the first True in valid_g[lo+1:hi].
"""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 75.0  # safety margin before the 90 s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# ---------- precompute primes up to the max prime ----------
_max_val = max(hi for lo, hi in P['gaps'])

def _primes_upto(n):
    """Return list of all primes <= n using a simple sieve."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

_primes = _primes_upto(_max_val)

# ---------- build valid-witness boolean arrays per distinct gap ----------
# For each distinct gap g, valid_g[m] = True iff no prime < g divides m.
# We only need m in the range of interest, but building over [0, _max_val]
# is cheap enough (max_val ~ 10289, a handful of distinct gaps).

_gaps_set = sorted(set(hi - lo for lo, hi in P['gaps']))
_valid_cache = {}

for g in _gaps_set:
    # primes strictly less than g
    small_primes = [p for p in _primes if p < g]
    # start with all True
    arr = np.ones(_max_val + 1, dtype=bool)
    arr[0] = False
    arr[1] = False
    for p in small_primes:
        # mark multiples of p as False
        arr[p::p] = False
    _valid_cache[g] = arr

# ---------- construct ----------
def construct():
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _START_TIME > _DEADLINE:
            break
        gap = hi - lo
        arr = _valid_cache[gap]
        # slice: indices lo+1 .. hi-1
        segment = arr[lo + 1:hi]
        # find first True
        idx = np.argmax(segment)
        if segment[idx]:
            result[i] = int(lo + 1 + idx)
    return result
