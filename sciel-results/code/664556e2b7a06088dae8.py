def construct():
    from math import factorial
    import time
    import sys
    sys.setrecursionlimit(10000)
    
    target = factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12!
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter: > 12
    valid_divs = sorted(set(d for d in divisors if d > 12))
    valid_divs_set = set(valid_divs)
    
    best = [target]  # fallback: single factor = 12!
    best_max = target
    
    # Precompute: for each divisor, store it for quick access
    # Key optimization: use branch and bound with geometric mean heuristic
    
    def dfs(remaining, current_factors, next_min, depth):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if current_factors:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        # If remaining > 1, we must pick at least one more factor.
        if next_min > remaining:
            return
        
        # Pruning: if remaining <= 12, can't be a valid factor
        if remaining <= 12:
            return
        
        # If remaining itself is a valid candidate and >= next_min, we can take it as last factor
        # But splitting might give smaller max, so we explore both
        
        # Try factors from valid_divs that divide remaining
        # Sort by closeness to geometric mean for better balance
        # But first, let's try ascending for simplicity, with strong pruning
        
        for f in valid_divs:
            if f < next_min:
                continue
            if f > remaining:
                break
            if f in current_factors:
                continue
            if remaining % f != 0:
                continue
            
            # If f >= best_max, no point continuing (all subsequent f are larger)
            if f >= best_max:
                break
            
            new_remaining = remaining // f
            
            if new_remaining == 1:
                # f is the last factor
                if f < best_max:
                    best_max = f
                    best = current_factors + [f]
                continue
            
            # new_remaining > 1, need more factors
            next_next_min = f + 1
            
            if new_remaining < next_next_min:
                continue
            if new_remaining <= 12:
                continue
            
            # Additional pruning: check if new_remaining can be decomposed
            # The smallest possible next factor is next_next_min
            # If new_remaining < next_next_min, we already handled that
            
            current_factors.append(f)
            dfs(new_remaining, current_factors, next_next_min, depth + 1)
            current_factors.pop()
            
            if time.time() > deadline:
                return
    
    dfs(target, [], 13, 0)
    
    return best
