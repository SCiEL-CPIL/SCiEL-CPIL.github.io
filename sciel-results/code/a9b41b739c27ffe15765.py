"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _compute_lcm(lst):
    r = 1
    for x in lst:
        r = _lcm(r, x)
    return r

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _compute_lcm(_DIVS)  # 30030
_N = len(_DIVS)
_MAX_VAL = 1000000

def _feasible(W):
    """Check if there exists a window of width W containing distinct multiples for all divisors.
    Returns (True, result_list) if found, else (False, None)."""
    for s in range(_LCM):
        vals = []
        valid = True
        for d in _DIVS:
            # Smallest multiple of d >= s
            rem = s % d
            if rem == 0:
                m = s
            else:
                m = s + (d - rem)
            if m > s + W:
                valid = False
                break
            vals.append(m)
        if not valid:
            continue
        # Check distinctness
        if len(set(vals)) != _N:
            continue
        # Check max value
        if max(vals) > _MAX_VAL:
            continue
        return (True, vals)
    return (False, None)

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill

    # Binary search on W
    lo = 1
    hi = 100  # upper bound; we know a solution exists well below this
    best_result = None
    best_W = None

    # First, find a valid upper bound
    while hi < _MAX_VAL:
        ok, res = _feasible(hi)
        if ok:
            best_result = res
            best_W = hi
            break
        hi *= 2

    if best_result is None:
        # Fallback
        return [int(d) for d in _DIVS]

    # Binary search for minimum W
    lo = max(1, best_W // 2) if best_W > 1 else 1
    # Actually, let's do a proper binary search
    lo = 1
    hi = best_W
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        ok, res = _feasible(mid)
        if ok:
            best_result = res
            best_W = mid
            hi = mid
        else:
            lo = mid + 1

    # Final check at lo
    if time.time() < deadline:
        ok, res = _feasible(lo)
        if ok:
            best_result = res
            best_W = lo

    if best_result is not None:
        return [int(x) for x in best_result]

    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END
