"""Rewritten program: find witnesses in every consecutive-prime gap."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061],
              [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091],
              [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111],
              [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151],
              [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177],
              [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223],
              [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259],
              [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_small(n):
    """Return list of primes up to n using a simple sieve."""
    if n < 2:
        return []
    flag = bytearray(b'\x01') * (n + 1)
    flag[0] = 0
    flag[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if flag[i]:
            for j in range(i * i, n + 1, i):
                flag[j] = 0
    return [k for k in range(2, n + 1) if flag[k]]


def _is_rough_scalar(m, forbidden):
    """Check whether m has no prime factor in forbidden (scalar)."""
    for p in forbidden:
        if m % p == 0:
            return False
    return True


def construct():
    t_start = time.perf_counter()
    limit = t_start + 50.0  # safety margin before 90s kill

    # Determine the maximum gap among the pinned pairs.
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _sieve_small(max_gap)

    # Group gaps by their size so we can cache forbidden-prime lists.
    forbidden_by_gap = {}
    for g in {hi - lo for lo, hi in P['gaps']}:
        forbidden_by_gap[g] = [p for p in small_primes if p < g]

    result = [None] * len(P['gaps'])

    # For each gap, find ALL rough candidates, then pick one.
    for idx, (lo, hi) in enumerate(P['gaps']):
        if time.perf_counter() > limit:
            break
        g = hi - lo
        forb = forbidden_by_gap[g]
        
        if not forb:
            # No forbidden primes, any number works
            result[idx] = int(lo + 1)
            continue
        
        # Generate all candidates in (lo, hi)
        candidates = np.arange(lo + 1, hi, dtype=np.int64)
        if len(candidates) == 0:
            continue
        
        # Check roughness using vectorized operations
        # For each forbidden prime, create a mask of candidates divisible by it
        rough_mask = np.ones(len(candidates), dtype=bool)
        for p in forb:
            rough_mask &= (candidates % p != 0)
        
        # Find indices where rough_mask is True
        rough_indices = np.where(rough_mask)[0]
        if len(rough_indices) > 0:
            # Pick the middle one for balance
            mid = len(rough_indices) // 2
            result[idx] = int(candidates[rough_indices[mid]])

    # Verify and fix any issues
    for idx, (lo, hi) in enumerate(P['gaps']):
        if time.perf_counter() > limit:
            break
        val = result[idx]
        if val is not None:
            g = hi - lo
            forb = forbidden_by_gap[g]
            if not (lo < val < hi) or not _is_rough_scalar(val, forb):
                # Fallback: re-scan scalar
                found = None
                for m in range(lo + 1, hi):
                    if _is_rough_scalar(m, forb):
                        found = m
                        break
                result[idx] = int(found) if found is not None else None

    # Final cleanup: ensure exact Python int types.
    out = []
    for v in result:
        if v is None:
            out.append(None)
        else:
            out.append(int(v))
    return out
# EVOLVE-BLOCK-END
