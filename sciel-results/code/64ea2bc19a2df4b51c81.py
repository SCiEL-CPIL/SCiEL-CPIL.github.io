"""Model-and-optimize rewrite for the consecutive-prime-gap witness task."""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 80.0  # explicit safety margin before the 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]


def _build_lpf_sieve(max_val):
    """Build an LPF (least prime factor) array up to max_val using NumPy.
    
    Returns a NumPy array where lpf[i] is the least prime factor of i,
    or 0 if i is 0 or 1, or i itself if i is prime.
    """
    lpf = np.arange(max_val + 1, dtype=np.int64)
    limit = int(math.isqrt(max_val))
    for i in range(2, limit + 1):
        if lpf[i] == i:  # i is prime
            start = i * i
            step = i
            indices = np.arange(start, max_val + 1, step)
            mask = lpf[indices] == indices
            lpf[indices[mask]] = i
    return lpf


def _is_prime_trial(n):
    """Trial division primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _lpf_trial(n):
    """Compute least prime factor via trial division."""
    if n < 2:
        return 0
    if n % 2 == 0:
        return 2
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return i
    return n  # n is prime


def _verify_witness(lo, hi, gap, m):
    """Verify that m is a valid witness: lo < m < hi and LPF(m) >= gap."""
    if not (lo < m < hi):
        return False
    if not isinstance(m, int) or isinstance(m, bool):
        return False
    lpf_val = _lpf_trial(m)
    return lpf_val >= gap


# Build LPF sieve up to the maximum prime in our pairs
_max_prime = max(hi for lo, hi in P['gaps'])
_lpf = _build_lpf_sieve(_max_prime)

# Verify LPF sieve correctness on a sample
def _verify_lpf():
    """Spot-check LPF values against trial division."""
    for val in [10007, 10009, 10037, 10273, 10289, 10272, 10288]:
        if val <= _max_prime:
            sieve_val = int(_lpf[val])
            trial_val = _lpf_trial(val)
            if sieve_val != trial_val:
                # Fall back to trial division for this value
                _lpf[val] = trial_val
    return True

_verify_lpf()


# Precompute witness for each gap using vectorized search
# For each pair, find ANY valid witness (not just the first)
_precomputed_witnesses = []
for (lo, hi) in P['gaps']:
    gap = hi - lo
    if gap <= 2:
        # Only one interior number: lo+1. LPF(lo+1) must be >= 2.
        # lo+1 is even (lo is odd prime > 2), so LPF = 2 >= 2. Always valid.
        _precomputed_witnesses.append(lo + 1)
    else:
        # Scan ALL interior numbers with trial division to be safe
        # Gaps are small (< 30), so at most ~29 interior numbers
        found = None
        for m in range(lo + 1, hi):
            if _lpf_trial(m) >= gap:
                found = m
                break
        _precomputed_witnesses.append(found)


def _find_witness(lo, hi, gap):
    """Find m in (lo, hi) with LPF(m) >= gap, or None if none exists.
    
    Uses precomputed LPF array for vectorized lookup, with trial division fallback.
    """
    if gap <= 2:
        return lo + 1
    interior_lpf = _lpf[lo + 1:hi]
    valid = np.where(interior_lpf >= gap)[0]
    if len(valid) > 0:
        candidate = int(lo + 1 + valid[0])
        if _lpf_trial(candidate) >= gap:
            return candidate
    # Fallback: scan with trial division
    for m in range(lo + 1, hi):
        if _lpf_trial(m) >= gap:
            return m
    return None


def construct():
    if time.time() - _START_TIME > _DEADLINE:
        # Return best effort: all gap-2 witnesses
        result = []
        for (lo, hi) in P['gaps']:
            gap = hi - lo
            if gap <= 2:
                result.append(lo + 1)
            else:
                result.append(None)
        return result
    
    result = []
    for i, w in enumerate(_precomputed_witnesses):
        if w is not None:
            # Final verification pass
            lo, hi = P['gaps'][i]
            gap = hi - lo
            if _verify_witness(lo, hi, gap, w):
                result.append(int(w))
            else:
                result.append(None)
        else:
            result.append(None)
    return result
