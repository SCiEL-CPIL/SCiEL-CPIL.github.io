"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = factorial(12)
    
    # Generate all divisors of N greater than 12
    divisors = []
    i = 1
    # Only need to check up to sqrt(N) for pairs, but we need all divisors
    # Actually, iterating 1 to N is too slow.
    # 12! is ~4.8e8. sqrt is ~22000.
    # We can iterate i from 1 to int(sqrt(N))
    # If N % i == 0, add i and N//i
    import math
    limit = int(math.isqrt(N))
    for i in range(1, limit + 1):
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            other = N // i
            if other > 12 and other != i:
                divisors.append(other)
    
    # Remove duplicates and sort
    divisors = sorted(list(set(divisors)))
    
    best_max = float('inf')
    best_factors = []
    
    # Backtracking search
    # We want to find a set of distinct factors from divisors
    # such that their product is N.
    # To minimize the max factor, we search for solutions with increasing max factor.
    # By iterating candidates in sorted order and pruning based on best_max,
    # we effectively find the optimal solution.
    
    def search(remaining, start_idx, current_factors):
        nonlocal best_max, best_factors
        
        # Check time
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if current_factors:
                current_max = max(current_factors)
                if current_max < best_max:
                    best_max = current_max
                    best_factors = current_factors[:]
            return
        
        # If we have more than 12 factors, stop (though distinct divisors of 12! 
        # that multiply to 12! cannot have too many small factors, 
        # but the constraint is <= 12 factors)
        if len(current_factors) >= 12:
            return
            
        # Pruning: The smallest possible next factor is divisors[start_idx]
        # If the product of current factors * (smallest possible next factor) > remaining, 
        # it's not useful, but we are dividing remaining.
        
        # Iterate through possible next factors
        for i in range(start_idx, len(divisors)):
            f = divisors[i]
            
            # If f is greater than or equal to the best max found so far, 
            # we can't improve the objective, so we stop.
            if f >= best_max:
                break
                
            # If f > remaining, no point (since factors are >= 1, but actually f must divide remaining)
            if f > remaining:
                break
                
            if remaining % f == 0:
                current_factors.append(f)
                search(remaining // f, i + 1, current_factors)
                current_factors.pop()
                
    search(N, 0, [])
    
    # If no solution found (should not happen), return the trivial one
    if not best_factors:
        return [N]
        
    return best_factors

# EVOLVE-BLOCK-END
