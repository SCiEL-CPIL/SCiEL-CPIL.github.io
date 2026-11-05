def construct():
    from math import factorial
    import time
    
    target = factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # All divisors of 12! that are > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Generate all divisors
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter: > 12, and < target (since we need at least one more factor)
    valid_divs = sorted(set(d for d in divisors if 12 < d < target))
    
    best = [target]  # fallback: single factor = 12!
    best_max = target
    
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
        
        if not current_factors and remaining == target:
            pass  # start
        
        # We need to pick a factor f such that:
        # f >= next_min, f divides remaining, f > 12, f not in current_factors
        # And remaining/f can be decomposed into distinct factors >= f+1 (or just >= next_min+1)
        
        # Bound: the minimum product of (max_factors - depth) distinct integers starting from next_min
        # If remaining < that minimum, we can't fill all slots, but we don't have to fill all slots
        # Actually, we just need remaining to be decomposable
        
        # Simple bound: if remaining < next_min, we can't pick any factor (since factor >= next_min and factor <= remaining)
        if remaining < next_min:
            return
        
        # Try factors from next_min up to remaining
        # But we need f to divide remaining
        # Iterate through divisors of remaining that are >= next_min
        
        # Find all divisors of remaining that are >= next_min
        # To be efficient, iterate through valid_divs and check if they divide remaining
        # But that's slow. Better: factor remaining and generate its divisors.
        
        # Alternative: just iterate through valid_divs (sorted) and check divisibility
        # valid_divs has at most ~790 elements, and we check each for divisibility
        
        for f in valid_divs:
            if f < next_min:
                continue
            if f > remaining:
                break
            if f in current_factors:
                continue
            if remaining % f != 0:
                continue
            
            # Pruning: remaining/f must be decomposable
            new_remaining = remaining // f
            
            # If new_remaining == 1, we're done (handled at top)
            # If new_remaining > 1, we need more factors
            # The next factor must be > f (to keep them distinct and in order)
            # Actually, next factor must be >= f+1, but also must be a divisor of new_remaining
            
            # Bound: if new_remaining > 1, the next factor is at least max(next_min, f+1)
            # But we also need new_remaining >= that next factor
            # If new_remaining < max(f+1, 13) and new_remaining > 1, we can't proceed
            next_next_min = max(f + 1, 13)
            if new_remaining > 1 and new_remaining < next_next_min:
                continue
            
            # Additional bound: the maximum number of factors we can still add
            # Each subsequent factor must be >= next_next_min, next_next_min+1, etc.
            # So the minimum product of m more factors is next_next_min * (next_next_min+1) * ...
            # If new_remaining < minimum product of 1 factor (which is next_next_min), and new_remaining > 1, skip
            # This is already covered above
            
            current_factors.append(f)
            dfs(new_remaining, current_factors, next_next_min, depth + 1)
            current_factors.pop()
            
            if time.time() > deadline:
                return
        
    dfs(target, [], 13, 0)
    
    return best
