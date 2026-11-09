"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin under 90s wall-clock
    
    # Greedy Egyptian fraction with backtracking, using exact rationals.
    # We want to represent 1 as a sum of distinct unit fractions 1/d_i,
    # each d_i in [5, 10_000_000], minimizing the number of terms.
    
    # Known small representations for reference (all denominators >= 5):
    # 1 = 1/5 + 1/6 + 1/8 + 1/12 + 1/24 + 1/30  ? No, sum is not 1.
    # Let's not hardcode; let the search find it.
    
    best_solution = None
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def dfs(remaining, last_d, used, depth, current_terms):
        nonlocal best_solution
        if time_left() < 0.5:
            return None
        
        if remaining == 0:
            return list(current_terms)
        
        if depth > 256:  # cap on terms
            return None
        
        if remaining < 0:
            return None
        
        # We need 1/d <= remaining, so d >= ceil(1/remaining)
        # Also d > last_d to ensure distinctness and increasing order
        # And d >= 5
        
        rem = remaining
        # d_min = max(5, last_d + 1, ceil(1/rem))
        # But rem is a Fraction. 1/rem is its reciprocal.
        one_over_rem = Fraction(1, 1) / rem
        d_min_from_rem = one_over_rem.numerator // one_over_rem.denominator
        if one_over_rem.numerator % one_over_rem.denominator != 0:
            d_min_from_rem += 1
        
        d_min = max(5, last_d + 1, d_min_from_rem)
        
        # Also, if we have k terms left to find, the largest possible sum
        # with terms starting from d_min is roughly bounded.
        # Simple bound: if 1/d_min + 1/(d_min+1) + ... is too small, prune.
        # But for now, let's just iterate.
        
        # Upper bound on d: if we can use at most (256 - depth) more terms,
        # and the smallest remaining term is 1/d, then 1/d <= remaining,
        # which gives d >= 1/remaining (already used).
        # Also, we can't have d > 10_000_000.
        d_max = 10_000_000
        
        # Additional pruning: if remaining * d_min < 1, then 1/d_min > remaining,
        # but we already ensured d_min >= ceil(1/remaining), so 1/d_min <= remaining.
        
        # Try d from d_min upwards, but limit iterations to avoid slow searches
        # when the gap is large.
        max_iter_per_step = 100000
        count = 0
        for d in range(d_min, d_max + 1):
            if count >= max_iter_per_step and (d - d_min) > 1000:
                break
            count += 1
            
            if time_left() < 0.5:
                return None
            
            # Check distinctness (d > last_d already ensures no duplicates in sorted order)
            # But we also need d not in used set for safety
            if d in used:
                continue
            
            if 1 / d > remaining:
                continue
            if 1 / d == remaining:
                # Exact match
                sol = current_terms + [d]
                if best_solution is None or len(sol) < len(best_solution):
                    best_solution = sol
                return sol
            
            # Recursive step
            new_remaining = remaining - Fraction(1, d)
            used.add(d)
            result = dfs(new_remaining, d, used, depth + 1, current_terms + [d])
            used.discard(d)
            
            if result is not None:
                if best_solution is None or len(result) < len(best_solution):
                    best_solution = result
                return result
    
    # Start the search
    result = dfs(Fraction(1, 1), 0, set(), 0, [])
    
    if result is not None:
        return result
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the original DP approach if search fails or times out
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
