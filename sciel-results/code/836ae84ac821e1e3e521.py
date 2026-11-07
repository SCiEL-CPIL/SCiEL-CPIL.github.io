import time
import numpy as np

_start_time = time.time()
_DEADLINE = 85.0

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(n):
    """Return list of primes up to n (inclusive)."""
    if n < 2:
        return []
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]

# Global LPF computation for the entire range
_lo_global = min(lo for lo, hi in P['gaps'])
_hi_global = max(hi for lo, hi in P['gaps'])
_range_start = _lo_global + 1  # first interior point
_range_end = _hi_global - 1    # last interior point
_range_len = _range_end - _range_start + 1

def _compute_lpf():
    """Compute least prime factor for every integer in [_range_start, _range_end].
    
    Returns an array lpf where lpf[i] is the least prime factor of (_range_start + i),
    or 0 if the number is prime (no prime factor <= sqrt(n), we treat prime as having
    LPF = itself which is certainly >= any gap).
    """
    # lpf[i] = 0 means "not yet assigned" -> will be set to the number itself if prime
    lpf = np.zeros(_range_len, dtype=np.int64)
    
    # Sieve: for each prime p, mark multiples in our range
    # We need primes up to sqrt(_hi_global) to find all LPFs
    sqrt_max = int(_hi_global**0.5) + 1
    primes = _sieve_primes(sqrt_max)
    
    for p in primes:
        # Find first multiple of p in [_range_start, _range_end]
        first = ((_range_start + p - 1) // p) * p
        if first > _range_end:
            break
        # Mark all multiples of p that haven't been marked yet
        idx_start = first - _range_start
        for idx in range(idx_start, _range_len, p):
            if lpf[idx] == 0:
                lpf[idx] = p
    
    # For unmarked entries, the number is prime (or 1, but our range starts > 10000)
    # Set LPF to the number itself (which is > any gap)
    for i in range(_range_len):
        if lpf[i] == 0:
            lpf[i] = _range_start + i
    
    return lpf

_lpf_array = _compute_lpf()

def _find_witness(lo, hi):
    """Find an integer m in (lo, hi) whose least prime factor is >= gap = hi - lo.
    
    Uses precomputed global LPF array.
    """
    gap = hi - lo
    # Interior range: [lo+1, hi-1]
    start_idx = (lo + 1) - _range_start
    end_idx = (hi - 1) - _range_start
    
    if start_idx < 0 or end_idx >= _range_len:
        return None
    if start_idx > end_idx:
        return None
    
    # Get LPF values for this interval
    interval_lpf = _lpf_array[start_idx:end_idx + 1]
    
    # Find indices where LPF >= gap
    valid_mask = interval_lpf >= gap
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return None
    
    # Pick the one with the highest LPF for robustness
    best_idx = valid_indices[np.argmax(interval_lpf[valid_indices])]
    return int(lo + 1 + best_idx)

def construct():
    global _start_time, _DEADLINE
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _start_time > _DEADLINE:
            break
        w = _find_witness(lo, hi)
        if w is not None:
            result[i] = w
    
    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
