"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    target = math.factorial(12)
    best_max = float('inf')
    best_factors = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def dfs(rem, last, factors):
        nonlocal best_max, best_factors
        
        # Check time limit
        if time.time() > deadline:
            return

        if rem == 1:
            if factors:
                curr_max = max(factors)
                if curr_max < best_max:
                    best_max = curr_max
                    best_factors = list(factors)
            return
        
        if len(factors) >= 12:
            return
            
        # The next factor must be > last and > 12
        start = max(last + 1, 13)
        
        # Case 1: The remaining value is the last factor
        # It must be distinct (so > last) and > 12
        # Also, to improve best_max, it should be < best_max
        if rem > last and rem > 12 and rem < best_max:
            dfs(1, rem, factors + [rem])
            
        # Case 2: Find an intermediate factor d
        # d must be a divisor of rem, d > last, d > 12
        # d should be <= sqrt(rem) to avoid duplicate permutations (since we enforce order)
        # Actually, if we pick d, the next factors will be > d.
        # If d > sqrt(rem), then rem//d < d, which violates the increasing order for the next step 
        # unless rem//d is 1 (handled in Case 1) or we stop.
        # So for intermediate steps, d <= sqrt(rem) is sufficient.
        
        limit = int(math.isqrt(rem))
        
        for d in range(start, limit + 1):
            if d >= best_max:
                break
            if rem % d == 0:
                dfs(rem // d, d, factors + [d])

    dfs(target, 0, [])
    
    if not best_factors:
        # Fallback: single factor
        return [target]
        
    return best_factors
# EVOLVE-BLOCK-END
