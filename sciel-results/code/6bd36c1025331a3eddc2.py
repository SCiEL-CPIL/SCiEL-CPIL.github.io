"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import factorial, gcd, prod
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600
    MIN_VAL = 13  # all factors must be > 12
    MAX_FACTORS = 12
    
    # Generate all divisors of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        for d in divisors:
            for i in range(e + 1):
                new_divisors.append(d * (p ** i))
        divisors = new_divisors
    
    # Filter divisors > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # For efficient lookup
    divisor_set = set(valid_divisors)
    
    best_solution = None
    best_max = N  # single factor case [N]
    
    # Try to find factorizations with k factors, k from MAX_FACTORS down to 2
    # For each k, we want k distinct divisors > 12 with product N, minimizing the max
    
    # Key insight: for k factors, the geometric mean is N^(1/k)
    # We want factors close to this value for balance
    
    # Use a recursive search with pruning
    def dfs(remaining, min_idx, current_factors, depth, max_depth, current_max):
        nonlocal best_solution, best_max
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if depth == max_depth:
            if remaining == 1:
                if len(current_factors) == max_depth:
                    max_f = max(current_factors)
                    if max_f < best_max:
                        best_max = max_f
                        best_solution = current_factors[:]
            return
        
        if remaining == 1:
            # We have fewer factors than max_depth, but that's fine
            # Actually we need exactly max_depth factors
            return
        
        # Pruning: if current_max is already >= best_max, no improvement possible
        # But we might still find a solution with same max but... no, we want strictly less
        # Actually we want to find any solution, then minimize
        # Better pruning: if the minimum possible max for remaining factors >= best_max, skip
        # The minimum possible max for (max_depth - depth) factors with product 'remaining'
        # is at least remaining^(1/(max_depth-depth))
        m_remaining = max_depth - depth
        if m_remaining > 0:
            min_possible_max = int(remaining ** (1.0 / m_remaining)) + 1
            if current_max >= best_max and min_possible_max >= best_max:
                return
        
        # We need to pick (max_depth - depth) more factors, all > min_factor (value-wise)
        # and distinct from current_factors
        # The next factor must be at least min_factor (value), and we search from min_idx in valid_divisors
        
        # Upper bound on next factor: if we pick factor f, then remaining/f must be
        # factorable into (m_remaining - 1) factors all > f
        # So remaining/f >= (f+1)(f+2)...(f+m_remaining-2) approximately
        # More precisely, the minimum product of (m_remaining-1) distinct integers > f
        # is (f+1)(f+2)...(f+m_remaining-2) but we need them to be divisors
        
        # Simpler bound: f^(m_remaining) <= remaining, so f <= remaining^(1/m_remaining)
        max_f_possible = int(remaining ** (1.0 / m_remaining)) + 2
        
        # Also, f must be at least the smallest unused divisor > min_factor
        # And f must divide remaining
        
        # Iterate through valid_divisors from min_idx, but only those that:
        # 1. Divide remaining
        # 2. Are > min_factor (value)
        # 3. Are <= max_f_possible
        # 4. Are not in current_factors
        
        # To be efficient, we can limit the range
        # The next factor f must satisfy:
        # f > min_factor_value (where min_factor_value is the value of the last factor chosen)
        # Actually, we track min_idx as the index in valid_divisors, so the value is valid_divisors[min_idx]
        
        min_factor_value = valid_divisors[min_idx] if min_idx < len(valid_divisors) else 0
        
        # We need f > min_factor_value for distinctness and ordering
        # But actually, we just need f not in current_factors and f > 12
        # The min_idx ensures we don't revisit smaller divisors
        
        # Let's iterate from min_idx to the end, but break when f > max_f_possible
        for i in range(min_idx, len(valid_divisors)):
            if time.time() > deadline:
                return
            
            f = valid_divisors[i]
            
            if f > max_f_possible:
                break
            
            if remaining % f != 0:
                continue
            
            if f in current_factors:
                continue
            
            # Additional pruning: remaining/f must be factorable into (m_remaining-1) factors > f
            # Minimum product of (m_remaining-1) distinct integers > f is (f+1)(f+2)...(f+m_remaining-2)
            # But they must be divisors of remaining/f, so this is a lower bound
            if m_remaining - 1 > 0:
                min_prod = 1
                for j in range(1, m_remaining):
                    min_prod *= (f + j)
                    if min_prod > remaining // f:
                        break
                if min_prod > remaining // f:
                    # Even the smallest possible remaining factors are too large
                    # Since f is increasing, this will only get worse
                    break
            
            new_factors = current_factors + [f]
            dfs(remaining // f, i + 1, new_factors, depth + 1, max_depth, max(current_max, f))
    
    # Try different numbers of factors, starting with more (likely to give smaller max)
    for k in range(MAX_FACTORS, 1, -1):
        if time.time() > deadline:
            break
        
        # Lower bound check: can we even have k factors?
        # The minimum product of k distinct integers > 12 is 13*14*...*13+k-1
        min_possible = 1
        for i in range(k):
            min_possible *= (13 + i)
        
        if min_possible > N:
            continue
        
        # Theoretical minimum max for k factors is at least N^(1/k)
        lower_bound = int(N ** (1.0 / k)) + 1
        
        # If we already have a solution with max <= lower_bound, we can't improve for this k
        if best_solution is not None and best_max <= lower_bound:
            break
        
        # Start the DFS with k factors
        dfs(N, 0, [], 0, k, 0)
        
        # Early termination if we found a very good solution
        if best_solution and best_max <= lower_bound + 2:
            break
    
    if best_solution is None:
        return [N]
    
    return best_solution

# EVOLVE-BLOCK-END
