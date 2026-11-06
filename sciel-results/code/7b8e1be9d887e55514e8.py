"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 12
    target = math.factorial(N)  # 479001600
    min_factor = N + 1  # 13
    
    # We want 1..12 distinct integers all > 12, product = target
    # Minimize the largest factor.
    # 
    # Strategy: DFS over divisors in increasing order.
    # State: (remaining_product, min_allowed_factor, current_factors_list, current_max)
    # We prune if:
    #   - current_max >= best_largest found so far
    #   - remaining_product < 1 (impossible)
    #   - remaining_product == 1: we've used all factors; update best if current_max < best
    #   - Number of factors used >= 12: if remaining != 1, dead end
    #   - The smallest possible distinct factors from min_allowed_factor upward cannot 
    #     multiply to remaining_product (lower bound check)
    #   - remaining_product must be composed of factors >= min_allowed_factor, all distinct
    #
    # Key insight: since factors must be distinct and > 12, and product is fixed,
    # the number of factors is bounded. 12! = 479001600.
    # 
    # Let's think about what's achievable. The product is 479001600.
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7 * 11
    # 
    # We need to partition these primes into distinct integers > 12.
    # 
    # Best approach: DFS choosing next factor from divisors of remaining_product,
    # in increasing order, starting from max(min_allowed_factor, current_last+1).
    # 
    # To make this fast, we precompute all divisors of target and filter those > 12.
    # But we need divisors of sub-products too. Better: at each step, find divisors of remaining.
    
    # Precompute: for efficiency, we'll generate divisors of a number on the fly
    # but cache them.
    
    divisor_cache = {}
    
    def get_divisors(n):
        if n in divisor_cache:
            return divisor_cache[n]
        divs = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        divisor_cache[n] = sorted(divs)
        return divisor_cache[n]
    
    best_largest = [target]  # mutable for closure
    best_factors = [target]
    
    def dfs(remaining, min_allowed, factors_so_far, current_max):
        if time.time() > deadline:
            return
        
        # Prune: if current_max already >= best, no improvement possible
        if current_max >= best_largest[0]:
            return
        
        # Base case
        if remaining == 1:
            if factors_so_far:
                if current_max < best_largest[0]:
                    best_largest[0] = current_max
                    best_factors[:] = factors_so_far[:]
            return
        
        # If we've used 12 factors already and remaining != 1, dead end
        if len(factors_so_far) >= 12:
            return
        
        # Lower bound: the minimum product we can form with distinct factors 
        # starting from min_allowed is min_allowed * (min_allowed+1) * ... 
        # But we can stop early. Let's check if even the smallest possible
        # single factor (min_allowed) can divide remaining, and if remaining/min_allowed
        # is achievable with remaining factors.
        
        # Quick prune: remaining must be >= min_allowed (at least one more factor needed)
        if remaining < min_allowed:
            return
        
        # Find divisors of remaining that are >= min_allowed
        divs = get_divisors(remaining)
        
        for d in divs:
            if d < min_allowed:
                continue
            if d >= best_largest[0]:
                # Since divisors are sorted, all subsequent will be >= best
                break
            
            # Check: can the quotient remaining//d be formed by distinct factors
            # all > d (i.e., >= d+1)?
            quotient = remaining // d
            
            # If quotient == 1, we're done
            if quotient == 1:
                new_max = max(current_max, d)
                if new_max < best_largest[0]:
                    best_largest[0] = new_max
                    best_factors[:] = factors_so_far + [d]
                continue
            
            # Lower bound check for quotient: the smallest distinct factors we could
            # use are d+1, d+2, ... Let's see if product of first k such factors
            # can reach quotient. At minimum, quotient must be >= d+1.
            if quotient < d + 1:
                continue
            
            # Additional prune: the minimum product of distinct factors starting from d+1
            # that could equal quotient. If even (d+1) doesn't divide quotient, skip.
            # Actually we just recurse; the recursion will handle it.
            # But let's add a stronger prune: 
            # The remaining factors must all be > d, distinct.
            # Minimum possible product for one more factor is d+1.
            # For two more factors: (d+1)*(d+2), etc.
            # If quotient < (d+1), skip (already checked).
            
            # Upper bound prune: if d * (d+1) * (d+2) * ... exceeds quotient for 
            # the minimum number of factors needed... this is complex.
            # Let's just recurse with the constraint that next factor > d.
            
            new_max = max(current_max, d)
            if new_max >= best_largest[0]:
                continue
            
            factors_so_far.append(d)
            dfs(quotient, d + 1, factors_so_far, new_max)
            factors_so_far.pop()
    
    # Start DFS
    # Initial: remaining = target, min_allowed = 13, no factors yet, current_max = 0
    dfs(target, min_factor, [], 0)
    
    return best_factors[:]
# EVOLVE-BLOCK-END
