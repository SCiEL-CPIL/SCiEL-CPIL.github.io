"""Improved program: search for distinct multiples minimizing span."""
import time
import math

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
DEADLINE = time.time() + 75  # safety margin vs 90s wall-clock


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
    best_span = None
    best_reps = None
    
    # Strategy 1: Scan L over a range of starting points
    # The optimal L is likely near where the multiples cluster.
    # Try L from 1 up to some limit, but be smart about step size.
    # Since divisors are small, try L in steps of 1 up to a reasonable limit,
    # then try larger steps.
    
    # First, try a fine scan over a moderate range
    # The span we're looking for is likely small (tens or hundreds).
    # Let's try L from 1 to 50000 with step 1, then expand if needed.
    
    limit_fine = 50000
    for L in range(1, limit_fine + 1):
        if time.time() > DEADLINE:
            break
        result = _evaluate_window(L)
        if result is not None:
            span, reps = result
            if best_span is None or span < best_span:
                best_span = span
                best_reps = reps[:]
    
    # If we found a good solution, try to improve by scanning more
    # around promising areas or with different strategies
    if best_span is not None and best_span <= 200:
        # Try to find even better by scanning L more finely around
        # the region that produced the best result
        pass  # Already did fine scan
    
    # Strategy 2: If no solution found, try larger L values
    if best_span is None:
        # Try L up to MAXV with larger steps
        for L in range(1, MAXV + 1, 10):
            if time.time() > DEADLINE:
                break
            result = _evaluate_window(L)
            if result is not None:
                span, reps = result
                if best_span is None or span < best_span:
                    best_span = span
                    best_reps = reps[:]
    
    if best_reps is None:
        # Fallback: just return the divisors themselves (valid but not optimal)
        return DIVISORS[:]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
