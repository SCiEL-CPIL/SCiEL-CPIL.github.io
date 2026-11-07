"""Improved program: search for distinct multiples minimizing span."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)
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


def _try_window_with_adjustments(L, max_adjustments=5):
    """Try window at L, and if conflicts exist, try small adjustments."""
    # First try the basic window
    result = _evaluate_window(L)
    if result is not None:
        return result
    
    # If there are conflicts, try adjusting some values
    # Find which divisors have conflicts
    base_reps = []
    for d in DIVISORS:
        m = _smallest_multiple_at_least(d, L)
        if m > MAXV:
            return None
        base_reps.append(m)
    
    # Find conflicts
    from collections import Counter
    counts = Counter(base_reps)
    conflicted_values = {v for v, c in counts.items() if c > 1}
    
    if not conflicted_values:
        # No conflicts but still invalid? Shouldn't happen
        return None
    
    # Try incrementing some of the conflicting values
    # For each divisor with a conflicting value, try the next multiple
    indices = list(range(N))
    
    # Try all combinations of incrementing conflicting divisors
    conflicting_indices = [i for i in range(N) if base_reps[i] in conflicted_values]
    
    # Try subsets of conflicting indices to increment
    from itertools import combinations
    for r in range(1, len(conflicting_indices) + 1):
        for combo in combinations(conflicting_indices, r):
            reps = base_reps[:]
            valid = True
            for i in combo:
                d = DIVISORS[i]
                # Next multiple
                reps[i] = reps[i] + d
                if reps[i] > MAXV:
                    valid = False
                    break
            if not valid:
                continue
            if len(set(reps)) != len(reps):
                continue
            span = max(reps) - min(reps)
            return (span, reps)
    
    return None


def construct():
    best_span = float('inf')
    best_reps = None
    
    # Strategy 1: Scan L values that are multiples of at least one divisor
    # The optimal L should be close to one of the representative values
    # Try L values around multiples of each divisor
    
    # First, get a quick baseline
    for L in range(1, 1000):
        if time.time() > DEADLINE:
            break
        result = _evaluate_window(L)
        if result is not None:
            span, reps = result
            if span < best_span:
                best_span = span
                best_reps = reps[:]
    
    # Now do targeted search around promising L values
    # For each divisor, try L near multiples of that divisor
    for d in DIVISORS:
        # Try L near multiples of d
        # The span is at most around max(DIVISORS) = 39 ideally
        # So we need to check L in a range
        start_mult = max(1, 1000 // d)
        end_mult = min(MAXV // d, 100000)
        
        # Sample L values
        for mult in range(start_mult, end_mult + 1, max(1, d // 2)):
            if time.time() > DEADLINE:
                break
            L = mult * d
            if L > MAXV:
                break
            result = _try_window_with_adjustments(L)
            if result is not None:
                span, reps = result
                if span < best_span:
                    best_span = span
                    best_reps = reps[:]
    
    # Strategy 2: Try L values that are the minimum of the representatives
    # The minimum representative is a multiple of some divisor
    # So try L = m * d for various m and d
    for d in DIVISORS:
        for m in range(1, 2000):
            if time.time() > DEADLINE:
                break
            L = m * d
            if L > MAXV:
                break
            result = _try_window_with_adjustments(L)
            if result is not None:
                span, reps = result
                if span < best_span:
                    best_span = span
                    best_reps = reps[:]
    
    # Strategy 3: Local search - for each candidate, try small perturbations
    if best_reps is not None:
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            # Try perturbing each representative by +/- its divisor
            improved = False
            for i in range(N):
                d = DIVISORS[i]
                for delta in [-d, d]:
                    new_reps = best_reps[:]
                    new_reps[i] += delta
                    if new_reps[i] <= 0 or new_reps[i] > MAXV:
                        continue
                    if len(set(new_reps)) != len(new_reps):
                        continue
                    span = max(new_reps) - min(new_reps)
                    if span < best_span:
                        best_span = span
                        best_reps = new_reps
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    if best_reps is None:
        # Fallback: return the divisors themselves (valid but not optimal)
        return DIVISORS[:]
    
    # Ensure we return Python ints
    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
