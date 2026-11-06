"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _greedy_complete(remaining, min_d, max_d, max_terms):
    """Exact greedy completion: pick smallest d such that 1/d <= remaining, d >= min_d, d <= max_d.
    Returns list of denominators or None if it fails."""
    result = []
    r = remaining
    for _ in range(max_terms):
        if r == 0:
            return result
        if r < 0:
            return None
        # We need 1/d <= r  =>  d >= 1/r
        # d must be >= min_d and <= max_d
        # Also d must be an integer >= ceil(1/r)
        num, den = r.numerator, r.denominator
        # ceil(1/r) = ceil(den/num)
        d_min = (den + num - 1) // num
        if d_min < min_d:
            d_min = min_d
        if d_min > max_d:
            return None
        d = d_min
        result.append(d)
        r -= Fraction(1, d)
        min_d = d + 1  # distinct and increasing
        if r == 0:
            return result
        if len(result) >= max_terms:
            if r != 0:
                return None
    if r == 0:
        return result
    return None

def _dfs_complete(remaining, min_d, max_d, max_terms, best_max_d):
    """DFS to find a completion with largest denominator < best_max_d.
    Returns (list_of_denoms, largest_d) or (None, inf)."""
    if remaining == 0:
        return ([], 0)
    if max_terms == 0:
        return (None, float('inf'))
    
    num, den = remaining.numerator, remaining.denominator
    # d must satisfy: 1/d <= remaining  =>  d >= ceil(den/num)
    d_low = (den + num - 1) // num
    if d_low < min_d:
        d_low = min_d
    if d_low > max_d:
        return (None, float('inf'))
    
    # Upper bound on d: we need to be able to complete with (max_terms-1) more terms.
    # The largest possible sum of (max_terms-1) terms starting from d+1 is bounded.
    # For now, just try d from d_low up to some reasonable limit.
    # Also, we want to minimize the largest denominator, so we prefer smaller d's that lead to smaller max.
    
    best_result = None
    best_largest = float('inf')
    
    # Try d values. To minimize the largest denominator, we should try smaller d first
    # but also consider that picking a very small d might force large denominators later.
    # A good heuristic: try d from d_low to min(max_d, d_low + 1000) initially, then expand if needed.
    # But for correctness within time budget, let's try a range.
    
    # Upper bound for d in this step: 
    # If we pick d, remaining' = remaining - 1/d must be >= 0 and completable.
    # Also, to keep the largest denominator small, we might want d not too small.
    # Let's iterate d from d_low to min(max_d, d_low + 5000).
    
    d_upper = min(max_d, d_low + 5000)
    
    for d in range(d_low, d_upper + 1):
        if d < min_d or d > max_d:
            continue
        r_new = remaining - Fraction(1, d)
        if r_new < 0:
            continue
        if r_new == 0:
            if d < best_largest:
                best_largest = d
                best_result = [d]
            continue
        
        # Prune: if d >= best_largest, no point continuing (since d is increasing)
        if d >= best_largest:
            break
        
        sub_result, sub_largest = _dfs_complete(r_new, d + 1, max_d, max_terms - 1, best_largest)
        if sub_result is not None:
            total_largest = max(d, sub_largest)
            if total_largest < best_largest:
                best_largest = total_largest
                best_result = [d] + sub_result
    
    return (best_result, best_largest)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe within 90s limit
    
    target = Fraction(5, 31)
    
    # Strategy: Try different first denominators d1, then use greedy/DFS to complete.
    # The largest denominator in the greedy completion for a given d1 can be computed.
    # We want to minimize the max denominator.
    
    best_solution = None
    best_max_d = float('inf')
    
    # First, get a baseline using the original approach (d1=8)
    baseline = _greedy_complete(target - Fraction(1, 8), 9, 10000000, 255)
    if baseline:
        sol = [8] + baseline
        max_d = max(sol)
        best_solution = sol
        best_max_d = max_d
    
    # Now try other first denominators
    # d1 must be such that 1/d1 < target => d1 > 1/target = 31/5 = 6.2 => d1 >= 7
    # Also, the remaining r = target - 1/d1 must be positive.
    # Let's try d1 from 7 up to some limit.
    
    # For each d1, we can compute the greedy completion and its max denominator.
    # We can also try a limited DFS for better results.
    
    for d1 in range(7, 200):  # Try first denominators 7 to 199
        if time.time() > deadline - 5:
            break
        
        r = target - Fraction(1, d1)
        if r <= 0:
            continue
        
        # Quick greedy check
        greedy_sol = _greedy_complete(r, d1 + 1, best_max_d - 1 if best_max_d > d1 else 10000000, 255)
        if greedy_sol:
            sol = [d1] + greedy_sol
            max_d = max(sol)
            if max_d < best_max_d:
                best_max_d = max_d
                best_solution = sol
        
        # Try DFS for a better solution, but only if we have time
        if time.time() > deadline - 10:
            break
        
        # DFS with a tighter bound
        dfs_sol, dfs_max = _dfs_complete(r, d1 + 1, best_max_d - 1 if best_max_d > d1 else 10000000, 255, best_max_d)
        if dfs_sol is not None:
            sol = [d1] + dfs_sol
            max_d = max(sol)
            if max_d < best_max_d:
                best_max_d = max_d
                best_solution = sol
    
    # If we still have time, try more d1 values or deeper DFS
    if best_solution is None:
        # Fallback: use the original
        r = target - Fraction(1, 8)
        values = [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution

# EVOLVE-BLOCK-END
