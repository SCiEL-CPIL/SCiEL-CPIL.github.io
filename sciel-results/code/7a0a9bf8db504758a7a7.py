# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_complete(r, start_d, max_denom, used_set, max_terms=256):
    """Complete r using greedy, all denominators >= start_d, <= max_denom, distinct."""
    values = []
    r = Fraction(r)
    d_min = start_d
    while r > 0:
        if len(values) >= max_terms:
            return None
        if r.numerator == 0:
            break
        # d = ceil(1/r)
        # 1/d <= r => d >= 1/r
        # d = ceil(r.denominator / r.numerator)
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d < d_min:
            d = d_min
        if d > max_denom:
            return None
        if d in used_set:
            d += 1
            while d <= max_denom and d in used_set:
                d += 1
            if d > max_denom:
                return None
        values.append(d)
        used_set.add(d)
        r -= Fraction(1, d)
        d_min = d + 1
    return values

def _find_small_triplets(target, max_a, max_b, max_denom, deadline):
    """Find solutions of form 1/a + 1/b + 1/c = target with a < b < c <= max_denom."""
    best = None
    best_max = float('inf')
    for a in range(1, max_a + 1):
        if time.time() > deadline:
            break
        ra = target - Fraction(1, a)
        if ra <= 0:
            continue
        # ra = (5a - 31) / (31a)
        # We need 1/b + 1/c = ra
        # b must be > a
        # 1/b < ra => b > 1/ra
        b_lo = (ra.denominator + ra.numerator - 1) // ra.numerator
        if b_lo <= a:
            b_lo = a + 1
        # Also 1/b + 1/c = ra => 1/c = ra - 1/b. 
        # For c to be integer, ra - 1/b must be reciprocal of integer.
        # Iterate b
        for b in range(b_lo, max_b + 1):
            if time.time() > deadline:
                break
            rb = ra - Fraction(1, b)
            if rb <= 0:
                break
            if rb.numerator != 1:
                continue
            c = rb.denominator
            if c <= b:
                continue
            if c > max_denom:
                continue
            max_d = c
            if max_d < best_max:
                best_max = max_d
                best = [a, b, c]
    return best

def construct():
    start_time = time.time()
    # Use 50s as hard limit to ensure we return before 60s
    deadline = start_time + 50.0  

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')

    # Phase 0: Explicit search for small triplets
    # 5/31 approx 0.16. a >= 7.
    # Search a up to 100, b up to 2000. 
    # This is 100 * 2000 = 200k iterations max. Fast.
    triplet_sol = _find_small_triplets(target, 100, 2000, 10000000, deadline)
    if triplet_sol is not None:
        best_solution = triplet_sol
        best_max_denom = max(triplet_sol)

    # Phase 1: Greedy with various starting denominators
    # If we found a good triplet, we might not need extensive greedy search.
    # But let's try to beat the triplet max if it's not very small.
    min_d = 7
    max_d_upper = 10000000
    
    # Limit the number of greedy tries to avoid timeout
    # If best_max_denom is small, we can be more aggressive.
    # Otherwise, just find a valid solution.
    
    if best_solution is None:
        # No triplet found quickly, use greedy to get a valid solution
        for first_d in range(min_d, min_d + 100):
            if time.time() > deadline:
                break
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                break
            used = {first_d}
            completion = _greedy_complete(remainder, first_d + 1, max_d_upper, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms
    else:
        # We have a triplet. Try to improve it.
        # Try greedy completions that might yield smaller max?
        # Greedy usually yields larger max than optimal, but let's check a few.
        # Also, try DFS-like search with very small window to refine.
        
        # Try to find a solution with max < best_max_denom
        # We can do a bounded search for 3 or 4 terms.
        
        # Simple heuristic: try different first denominators and see if greedy beats current best
        search_start = max(min_d, best_solution[0] - 10)
        search_end = min(100, best_solution[0] + 50) # Don't go too high
        
        for first_d in range(search_start, search_end + 1):
            if time.time() > deadline:
                break
            if first_d in best_solution:
                continue
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                continue
            used = {first_d}
            # Try to complete with max allowed = best_max_denom - 1
            max_allowed = best_max_denom - 1
            if max_allowed < first_d + 1:
                continue
            completion = _greedy_complete(remainder, first_d + 1, max_allowed, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms

    if best_solution is None:
        # Fallback: simple greedy without max_denom constraint initially
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
            if time.time() > deadline:
                break
        best_solution = values

    return best_solution
# EVOLVE-BLOCK-END
