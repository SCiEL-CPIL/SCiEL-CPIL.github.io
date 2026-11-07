import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _smallest_multiple_at_least(d, L):
    """Return the smallest multiple of d that is >= L."""
    return ((L + d - 1) // d) * d

def _evaluate_window(L, divisors):
    """
    Given a lower bound L, compute the smallest multiple of each divisor >= L.
    Returns (max_val, min_val, reps) or None if any rep > MAX_VAL.
    """
    reps = []
    for d in divisors:
        r = _smallest_multiple_at_least(d, L)
        if r > MAX_VAL:
            return None
        reps.append(r)
    
    # Check distinctness
    if len(set(reps)) != len(reps):
        return None
    
    mx = max(reps)
    mn = min(reps)
    return (mx, mn, reps)

def _local_search_around(L, divisors, current_best_span, max_iterations=10000):
    """
    Try to improve upon current_best_span by searching for a better window
    starting near L. We vary L slightly and also try different orderings/assignments
    is not needed here because the assignment is fixed by the problem statement
    (reps[i] corresponds to divisors[i]).
    
    However, the 'window' approach assumes we pick L and then take the smallest multiple >= L.
    This might not be optimal if a larger multiple for one divisor allows a smaller max
    overall? No, because if we increase a rep, max can only stay same or increase, and min
    can only increase. Wait.
    
    Let's re-evaluate the strategy.
    The objective is min(max(reps) - min(reps)).
    
    Strategy 1: Sweep L.
    For a fixed L, the optimal choice for each divisor d_i is the smallest multiple of d_i that is >= L.
    Let this be r_i(L).
    Then span(L) = max(r_i(L)) - min(r_i(L)).
    Is it always optimal to pick the smallest multiple >= L?
    Suppose we pick L. If we increase r_j for some j, max might increase or stay same, min might increase or stay same.
    If min increases, span might decrease if max stays same.
    But if we change L, the set of valid reps changes.
    
    Actually, the set of all valid tuples (r_1, ..., r_n) is the Cartesian product of multiples.
    We want to minimize max(r) - min(r).
    Let M = max(r) and m = min(r). We want M - m to be small.
    This is equivalent to finding the smallest interval [m, M] that contains at least one multiple of each d_i.
    And the reps are those multiples.
    BUT, the reps must be distinct.
    
    So the problem is: Find the smallest interval [m, M] such that for each d_i, there exists a multiple k_i * d_i in [m, M], and all k_i * d_i are distinct.
    Then we can pick those multiples.
    
    This is a "covering" problem.
    Let's sweep the interval length S. Binary search on S?
    For a given S, check if there exists an interval of length S that covers all divisors with distinct multiples.
    
    Check(S):
    Is there an integer m such that for each i, there is a multiple of d_i in [m, m+S], and all chosen multiples are distinct?
    
    This is still complex due to distinctness.
    However, the divisors are relatively small and fixed.
    The LCM of the divisors is large, but we only care about local patterns.
    
    Let's stick to the sweep of L (the lower bound of the interval) but be smarter.
    The function span(L) = max(r_i(L)) - min(r_i(L)) is piecewise constant or step-like.
    Actually, as L increases by 1, each r_i(L) either stays the same or jumps to the next multiple.
    So r_i(L) is a step function.
    max(r_i(L)) and min(r_i(L)) are also step functions.
    The span changes only when some r_i(L) changes.
    The critical values of L are the multiples of the divisors.
    
    So we only need to check L values that are multiples of some divisor.
    Specifically, if L is in (m_k, m_{k+1}] where m_k are sorted multiples of all divisors, the set of r_i(L) is constant?
    No. If L is between two consecutive multiples of d_i, r_i(L) is the upper multiple.
    So r_i(L) changes exactly at the multiples of d_i.
    Therefore, the span(L) is constant on intervals between consecutive multiples of ANY divisor.
    So we only need to evaluate span(L) at L = each multiple of each divisor.
    
    How many such L are there?
    Sum(1000000 / d_i) ≈ 1000000 * (1/6 + 1/10 + ... ) ≈ 1000000 * 0.6 ≈ 600,000.
    This is manageable.
    
    Algorithm:
    1. Generate all multiples of each divisor up to MAX_VAL.
    2. Sort them.
    3. For each unique multiple L (in increasing order):
       a. Compute r_i = smallest multiple of d_i >= L.
       b. Check if all r_i <= MAX_VAL.
       c. Check if all r_i are distinct.
       d. Compute span = max(r_i) - min(r_i).
       e. Update best.
    
    We can optimize by breaking early if the theoretical minimum span is reached?
    The minimum possible span is 0 (impossible due to distinctness) or 1, etc.
    With 10 distinct integers, the min span is at least 9.
    
    Let's implement this sweep.
    
    To speed up, we can precompute the next multiple for each divisor for a given L?
    No, just use the formula.
    
    We can also stop the sweep if L > MAX_VAL - 9 (since we need at least 9 span).
    Actually, if L is too large, r_i might exceed MAX_VAL.
    
    Let's refine the loop.
    """
    pass # Replaced by main logic in construct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Generate all candidate L values (multiples of divisors)
    # We can do this efficiently
    candidates = set()
    for d in DIVISORS:
        for m in range(d, MAX_VAL + 1, d):
            candidates.add(m)
    
    # Sort candidates
    sorted_candidates = sorted(candidates)
    
    best_span = float('inf')
    best_reps = None
    
    # To speed up, we can skip L values that are "dominated"?
    # No, just iterate.
    
    # Optimization: 
    # If we process L in increasing order, r_i(L) are non-decreasing.
    # max(r_i) is non-decreasing.
    # min(r_i) is non-decreasing.
    # span = max - min.
    
    # We can break if min(r_i) > MAX_VAL - 9? No.
    # We can break if the current best_span is 9 (minimum possible for 10 distinct ints).
    
    for L in sorted_candidates:
        if time.time() > deadline:
            break
        
        # If we already found a span of 9, we can't do better.
        if best_span == 9:
            break
            
        # Compute reps
        reps = []
        valid = True
        for d in DIVISORS:
            r = _smallest_multiple_at_least(d, L)
            if r > MAX_VAL:
                valid = False
                break
            reps.append(r)
        
        if not valid:
            # Since L is increasing, if r > MAX_VAL for some d, 
            # it might still be valid for larger L? No, r only increases.
            # So if r > MAX_VAL, it will be > MAX_VAL for all larger L.
            # Wait, if L increases, r increases. So if r > MAX_VAL, we can break the entire loop?
            # Yes, because for any L' > L, r_i(L') >= r_i(L) > MAX_VAL.
            break
        
        # Check distinctness
        if len(set(reps)) != len(reps):
            continue
        
        mx = max(reps)
        mn = min(reps)
        span = mx - mn
        
        if span < best_span:
            best_span = span
            best_reps = reps[:]
            
    # If no valid solution found (should not happen), fallback to greedy
    if best_reps is None:
        # Simple fallback: just take the divisors themselves? 
        # They are distinct? 6,10,14,15,21,22,26,33,35,39. Yes.
        best_reps = DIVISORS[:]
        
    # Ensure types are correct
    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
