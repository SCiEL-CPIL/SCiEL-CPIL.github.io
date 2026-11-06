import time
from math import factorial
from functools import lru_cache

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill
    
    N = factorial(12)  # 479001600
    
    # Precompute all divisors of N
    divisors = set()
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
        i += 1
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # For each divisor, precompute which divisors of N/d are also > 12
    # This will be used in DFS
    
    best_max = N  # initial: single factor
    best_factors = [N]
    
    def dfs(remaining, min_factor, count, factors, max_allowed):
        nonlocal best_max, best_factors
        
        if time.time() > DEADLINE:
            return
        
        # If we've used all factors we want, check if remaining is 1
        if count >= 12:
            if remaining == 1:
                if max(factors) < best_max:
                    best_max = max(factors)
                    best_factors = factors[:]
            return
        
        # If remaining == 1, we have a valid solution
        if remaining == 1:
            if len(factors) >= 1:
                if max(factors) < best_max:
                    best_max = max(factors)
                    best_factors = factors[:]
            return
        
        # Pruning: the remaining product must be expressible as a product of
        # (12 - count) distinct integers, each >= min_factor
        # Lower bound: the smallest possible product with (12-count) distinct integers >= min_factor
        # is min_factor * (min_factor+1) * ... * (min_factor + (12-count-1))
        num_remaining = 12 - count
        if num_remaining > 0:
            # Check if remaining can be at least the product of the smallest num_remaining distinct ints >= min_factor
            prod_min = 1
            for j in range(num_remaining):
                prod_min *= (min_factor + j)
                if prod_min > remaining:
                    break
            if prod_min > remaining:
                return
        
        # Also, remaining must be >= min_factor (at least one more factor needed)
        if remaining < min_factor:
            return
        
        # Try each possible next factor
        for f in valid_divs:
            if f < min_factor:
                continue
            if f > max_allowed:
                break  # since valid_divs is sorted
            if remaining % f != 0:
                continue
            
            new_remaining = remaining // f
            factors.append(f)
            dfs(new_remaining, f + 1, count + 1, factors, max_allowed)
            factors.pop()
            
            if time.time() > DEADLINE:
                return
    
    # Start DFS: we can use 1 to 12 factors
    # To minimize the max, we should try to find solutions with smaller max
    # Let's do iterative deepening on max_allowed
    
    # First, get an initial bound: try with all factors up to some reasonable limit
    # Start with max_allowed = N (the trivial solution)
    dfs(N, 13, 0, [], N)
    
    if best_max == N:
        return [N]
    else:
        return best_factors
