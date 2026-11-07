"""Improved program: search for distinct multiples minimizing span."""
import time
import math
from itertools import product

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)


def _smallest_multiple_at_least(d, L):
    """Return smallest multiple of d that is >= L."""
    return ((L + d - 1) // d) * d


def _evaluate_window(L):
    """For a given lower bound L, compute smallest multiple of each divisor >= L.
    Return (span, reps) if all distinct and <= MAXV, else None."""
    reps = []
    for d in DIVISORS:
        m = _smallest_multiple_at_least(d, L)
        if m > MAXV:
            return None
        reps.append(m)
    if len(set(reps)) != len(reps):
        return None
    span = max(reps) - min(reps)
    return (span, reps)


def _feasible_span(S, L_min=1, L_max=None):
    """Check if there exists L in [L_min, L_max] such that all 10 divisors have
    a distinct multiple in [L, L+S]. Returns (L, reps) if found, else None."""
    if L_max is None:
        L_max = MAXV - S
    if L_max < L_min:
        return None
    # Critical L values: L such that L is a multiple of some d, or L+S is a multiple of some d
    # Also try L values near these critical points
    candidates = set()
    for d in DIVISORS:
        # L such that L % d == 0
        first = ((L_min + d - 1) // d) * d
        for L_cand in range(first, L_max + 1, d):
            candidates.add(L_cand)
        # L such that (L+S) % d == 0, i.e., L % d == (-S) % d
        rem = (-S) % d
        if rem == 0:
            first = ((L_min + d - 1) // d) * d
        else:
            first = L_min + (rem - L_min % d) % d
            if first < L_min:
                first += d
        for L_cand in range(first, L_max + 1, d):
            candidates.add(L_cand)
    
    # Sort and evaluate
    for L in sorted(candidates):
        if L < L_min or L > L_max:
            continue
        reps = []
        valid = True
        for d in DIVISORS:
            m = _smallest_multiple_at_least(d, L)
            if m > L + S or m > MAXV:
                valid = False
                break
            reps.append(m)
        if not valid:
            continue
        if len(set(reps)) != len(reps):
            continue
        return (L, reps)
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin vs 90s wall-clock
    
    # Initial solution: use the divisors themselves
    best_reps = DIVISORS[:]
    best_span = max(best_reps) - min(best_reps)
    
    # Binary search on span
    lo, hi = 9, best_span  # min span for 10 distinct ints is 9
    
    # First, verify hi is feasible (it should be since we have a solution)
    # Now binary search
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = _feasible_span(mid)
        if result is not None:
            L, reps = result
            span = max(reps) - min(reps)
            if span <= mid:
                hi = mid
                best_reps = reps[:]
                best_span = span
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # After binary search, hi is the minimum feasible span
    # Do a final check
    result = _feasible_span(hi)
    if result is not None:
        L, reps = result
        span = max(reps) - min(reps)
        if span < best_span:
            best_span = span
            best_reps = reps[:]
    
    return [int(x) for x in best_reps]

# EVOLVE-BLOCK-END
