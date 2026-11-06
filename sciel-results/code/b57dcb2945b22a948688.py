"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill
    
    N = factorial(12)  # 479001600
    
    # We need 1..12 distinct integers all > 12 with product exactly N.
    # Minimize the largest factor.
    
    # First, let's find all divisors of N that are > 12.
    # Then we search for subsets of these divisors whose product is N,
    # with all elements distinct and > 12, minimizing the max element.
    
    # Key insight: we want to minimize the maximum factor.
    # Let's do a DFS where we pick factors in increasing order,
    # and prune aggressively.
    
    # Generate all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            other = N // i
            if other != i and other > 12:
                divisors.append(other)
        i += 1
    divisors.sort()
    
    # Remove duplicates
    divisors = list(set(divisors))
    divisors.sort()
    
    # We want subsets of divisors (distinct) with product = N, minimizing max.
    # DFS approach: pick factors in increasing order to ensure distinctness.
    # At each step, we choose the next factor from divisors greater than the last chosen,
    # such that the remaining product can be formed by distinct factors all greater than the current one.
    
    best_max = N  # worst case: single factor N
    best_factors = [N]
    
    # Precompute: for a given remaining product R and minimum allowed factor min_f,
    # can we form R as a product of distinct factors all >= min_f (and > 12)?
    # This is hard to precompute, so we use pruning heuristics.
    
    # Heuristic pruning:
    # 1. If remaining product R < min_f, impossible (need at least one factor >= min_f)
    # 2. If R == 1, we're done
    # 3. If R is prime and R < min_f, impossible
    # 4. The number of factors we can still add is limited by log_2(R) roughly
    
    # Let's do DFS with the following strategy:
    # - Start with empty list, min_factor = 13 (since all must be > 12)
    # - At each step, try divisors d where d >= min_factor and d divides remaining
    # - Prune if d >= best_max (since we're minimizing max)
    # - Prune if remaining // d < min_next where min_next = d + 1 (next factor must be > d)
    #   Actually, next factor must be > d (distinct and increasing), so >= d+1
    # - Also prune if remaining // d is prime and < d+1
    
    # Better: use the fact that if we have k factors left to pick, each >= min_f,
    # their product is at least min_f * (min_f+1) * ... * (min_f+k-1)
    # So if R < product of smallest k possible factors, prune.
    
    # But we don't know k. Let's just try.
    
    # Sort divisors in increasing order for the DFS
    
    def dfs(remaining, min_f, current_factors):
        nonlocal best_max, best_factors
        if time.time() - t0 > DEADLINE:
            return
        
        if remaining == 1:
            if current_factors:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best_factors = list(current_factors)
            return
        
        # If remaining < min_f, we can't pick any more factors
        if remaining < min_f:
            return
        
        # If remaining is a divisor of N and remaining >= min_f and remaining < best_max,
        # we can take it as the last factor
        if remaining >= min_f and remaining < best_max and N % remaining == 0:
            # Check if remaining is in our divisor list (it should be if it divides N)
            # Take it
            mx = max(current_factors + [remaining]) if current_factors else remaining
            if mx < best_max:
                best_max = mx
                best_factors = list(current_factors) + [remaining]
        
        # Try to split remaining into two or more factors
        # Try each divisor d of remaining where d >= min_f
        # d must be < remaining (since we need at least one more factor after)
        # and remaining // d >= d + 1 (next factor must be > d)
        
        r = isqrt(remaining)
        for d in range(min_f, r + 1):
            if time.time() - t0 > DEADLINE:
                return
            if remaining % d != 0:
                continue
            other = remaining // d
            # other must be >= d + 1 for distinctness (since we go in increasing order)
            # Actually, other could be equal to d if we allow, but we need distinct factors.
            # Since we're picking in increasing order, other > d is required for the next pick.
            # But wait: we could have other == d only if we pick d twice, which is not allowed.
            # So we need other > d.
            if other <= d:
                continue
            # other must be >= d + 1
            # Also, other must be such that it can be further decomposed or is itself a valid factor
            # Pruning: if other < d + 1, skip (already checked)
            # If other >= best_max, skip since max would be at least other
            if other >= best_max:
                continue
            
            # Recurse: pick d, then continue with other, min_f = d + 1
            current_factors.append(d)
            dfs(other, d + 1, current_factors)
            current_factors.pop()
    
    dfs(N, 13, [])
    
    if best_factors:
        return best_factors
    else:
        return [N]
# EVOLVE-BLOCK-END
