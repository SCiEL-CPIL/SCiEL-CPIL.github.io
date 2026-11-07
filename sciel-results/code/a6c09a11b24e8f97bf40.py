"""Improved program: search for distinct multiples minimizing span."""
import time
import math

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
    # Check distinctness
    if len(set(reps)) != len(reps):
        return None
    span = max(reps) - min(reps)
    return (span, reps)


def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin vs 90s wall-clock
    
    best_span = None
    best_reps = None
    
    # Strategy:
    # 1. First, try the "greedy" approach: for each possible L, compute the smallest
    #    multiple of each divisor >= L, check if they're all distinct.
    # 2. The optimal L is likely near the minimum of the optimal solution.
    # 3. We can scan L values, but smartly.
    
    # Key insight: if we fix L, the best representative for divisor d is the smallest
    # multiple of d >= L. If any two are equal, we can try incrementing one of them
    # to the next multiple. But this gets complex.
    
    # Alternative: enumerate candidate windows. The span is max - min.
    # For a given span S, can we fit all 10 distinct multiples in [L, L+S]?
    # This is a feasibility check. We can binary search on S.
    
    # Let's do a binary search on the span S.
    # For a given S, check if there exists L such that for each divisor d,
    # there is a multiple of d in [L, L+S], and all 10 multiples are distinct.
    
    # Actually, let's think differently. The span is at most... well, let's just
    # do a smart search.
    
    # First, get a quick upper bound using the greedy method.
    # Try L = 1, 2, ..., and also try L values that are multiples of various numbers.
    
    # Let's start with a coarse scan, then refine.
    
    # Quick upper bound: just take the divisors themselves (they're all distinct and <= 39)
    # Wait, 6,10,14,15,21,22,26,33,35,39 are all distinct. Span = 39-6 = 33.
    initial_reps = DIVISORS[:]
    initial_span = max(initial_reps) - min(initial_reps)
    best_span = initial_span
    best_reps = initial_reps[:]
    
    # Now try to improve. The greedy window approach:
    # For a given L, the smallest multiple of d >= L is well-defined.
    # We want to find L such that all 10 smallest multiples are distinct.
    
    # Let's scan L in a smart way. The critical L values are those where
    # some multiple changes. But that's complex.
    
    # Instead, let's scan L from 1 to some limit, but skip intelligently.
    # Actually, the span is likely small (around 30-50), so let's try:
    # For each candidate span S starting from 1 upward, check if feasible.
    
    # But checking feasibility for a given S is also non-trivial.
    
    # Let's just do a direct search over L values, but be smart about which L to try.
    # The optimal solution has min(reps) = some value M. For that M, the greedy
    # assignment (smallest multiple >= M for each divisor) should give distinct values
    # if the optimal solution is "tight".
    
    # Let's try M values. M can be any integer from 1 to MAXV - max_divisor.
    # But we can't scan all. Let's try M values that are multiples of small numbers,
    # or M values near the divisors.
    
    # Better approach: iterate over possible minimum values. The minimum value in the
    # optimal solution is a multiple of one of the divisors. Let's try each divisor's
    # multiples as candidate minimums.
    
    # Actually, let's just scan L from 1 to 2000 first (quick), then expand if needed.
    
    # Phase 1: scan L from 1 to 5000
    L = 1
    while L <= 5000 and time.time() < deadline:
        result = _evaluate_window(L)
        if result is not None:
            span, reps = result
            if span < best_span:
                best_span = span
                best_reps = reps[:]
                if best_span == 1:  # minimum possible for 2 distinct... but we need 10 distinct
                    # Actually minimum span for 10 distinct integers is 9.
                    if best_span == 9:
                        break
        L += 1
    
    # Phase 2: if we haven't found span 9, try more L values.
    # The optimal L might be larger. Let's try L values that are multiples of the LCM
    # or near multiples of the divisors.
    
    # Actually, let's think about what span is achievable.
    # We need 10 distinct multiples in [L, L+S].
    # The densest packing would be consecutive integers, but they must be multiples
    # of specific divisors.
    
    # Let's try a different approach: for each pair of divisors, the gap between
    # their multiples matters.
    
    # Let's just continue scanning L further if time permits.
    L = 5001
    while L <= 50000 and time.time() < deadline:
        # Skip L values that are unlikely to be optimal.
        # The minimum of the solution is likely a multiple of one of the divisors.
        # So only try L that is a multiple of at least one divisor.
        is_multiple = any(L % d == 0 for d in DIVISORS)
        if is_multiple:
            result = _evaluate_window(L)
            if result is not None:
                span, reps = result
                if span < best_span:
                    best_span = span
                    best_reps = reps[:]
                    if best_span == 9:
                        break
        L += 1
    
    # Phase 3: try even larger L values, but only multiples of divisors.
    L = 50001
    while L <= 200000 and time.time() < deadline:
        is_multiple = any(L % d == 0 for d in DIVISORS)
        if is_multiple:
            result = _evaluate_window(L)
            if result is not None:
                span, reps = result
                if span < best_span:
                    best_span = span
                    best_reps = reps[:]
                    if best_span == 9:
                        break
        L += 1
    
    # Ensure we return a valid list of Python ints
    return [int(x) for x in best_reps]

# EVOLVE-BLOCK-END
