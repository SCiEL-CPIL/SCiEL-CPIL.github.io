def construct():
    from math import factorial
    import time
    import sys
    sys.setrecursionlimit(10000)
    
    target = factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
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
    
    # Filter: > 12, and < target (since we need at least one more factor if this isn't the last)
    # Actually, if it's the last factor, it can be target. But in DFS we handle remaining==1.
    # If we pick f = remaining, then new_remaining = 1.
    # So valid_divs should include everything > 12.
    valid_divs = sorted(set(d for d in divisors if d > 12))
    
    # Precompute index for quick lookup? Or just use set for "in current_factors"
    # current_factors is small (<=12), so list check is fine.
    
    best = [target]  # fallback: single factor = 12!
    best_max = target
    
    # Precompute powers of 2 for bounds?
    # Let's use a helper to check if remaining can be decomposed.
    # A strong prune: 
    # If remaining > 1, we need at least one factor f such that f > 12, f divides remaining, f >= next_min.
    # Also, the remaining product after picking f must be 1 or decomposable.
    
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
        # The next factor f must be >= next_min.
        # If next_min > remaining, we can't pick any factor (since f <= remaining).
        if next_min > remaining:
            return
        
        # Try factors from next_min up to remaining
        # Iterate through valid_divs (sorted)
        # Optimization: If we are close to deadline, maybe prioritize finding *any* solution with low max?
        # The current DFS finds solutions.
        
        # Pruning based on geometric mean?
        # If we pick f, remaining/f must be decomposable.
        
        # Iterate candidates
        # To minimize the largest factor, we want to balance factors.
        # However, DFS with sorted divisors explores small factors first.
        # This might lead to many small factors and one large one if not careful?
        # No, if we pick small factors, remaining stays large, forcing later factors to be large.
        # If we pick larger factors (closer to geometric mean), remaining shrinks faster.
        
        # Let's try iterating divisors in a specific order.
        # Standard DFS: ascending.
        # Maybe we should check if remaining itself is a valid candidate (i.e. remaining > 12 and remaining >= next_min).
        # If so, we can just add it and finish.
        # This is a "greedy" finish.
        
        # Let's stick to the iterative approach but add a check for the "single factor remaining" case early?
        # No, DFS handles it: if we pick f = remaining, new_remaining = 1, and we update best.
        
        # Optimization: 
        # If remaining is in valid_divs and remaining >= next_min and remaining > 12:
        # Then we can just take it as the last factor.
        # Is it better than splitting?
        # Splitting remaining into a*b where a,b > 12 and a,b distinct and > next_min...
        # If we split, max(a,b) < remaining.
        # So splitting is ALWAYS better for minimizing max, IF possible.
        # So we should prioritize splitting if possible.
        
        # How to check if splitting is possible?
        # Find any divisor d of remaining such that next_min <= d < remaining/d (so d < sqrt(remaining) roughly, or just d < remaining)
        # and remaining/d >= d+1 (distinctness) and remaining/d > 12.
        
        # Let's just run the DFS.
        
        for f in valid_divs:
            if f < next_min:
                continue
            if f > remaining:
                break
            if f in current_factors:
                continue
            if remaining % f != 0:
                continue
            
            new_remaining = remaining // f
            
            # If new_remaining == 1, this is a valid completion.
            # The max factor will be max(current_factors + [f]).
            # Since f >= next_min > previous factors, max is f.
            # So candidate_max = f.
            # If f < best_max, we can update.
            # But we also need to check if we can do better by NOT taking f as the last one?
            # No, if new_remaining == 1, we MUST take f as the last one.
            # So if new_remaining == 1:
            #   if f < best_max: update best
            #   continue (to next f)
            
            if new_remaining == 1:
                if f < best_max:
                    best_max = f
                    best = current_factors + [f]
                continue
            
            # If new_remaining > 1:
            # We need to decompose new_remaining.
            # The next factor must be > f (to keep distinct and sorted order? No, just distinct).
            # But in our DFS, we enforce non-decreasing order to avoid permutations.
            # So next factor must be >= f + 1?
            # Actually, if we enforce order f1 < f2 < ... < fk, then next_min becomes f + 1.
            # Wait, if we pick f=13, next can be 14.
            # If we pick f=13, next can be 13? No, distinct.
            # So next_min = f + 1 is correct for strictly increasing sequence.
            
            next_next_min = f + 1
            
            # Prune: Can we form new_remaining with factors >= next_next_min?
            # The smallest possible factor is next_next_min.
            # If new_remaining < next_next_min, we can't even pick the smallest allowed factor.
            # UNLESS new_remaining == 1 (handled above).
            if new_remaining < next_next_min:
                continue
            
            # Additional prune:
            # Is it possible to decompose new_remaining into distinct integers >= next_next_min?
            # This is a hard problem to check exactly fast.
            # But we can check if new_remaining has any divisor >= next_next_min and < new_remaining.
            # If not, and new_remaining > 12, then we are stuck unless new_remaining itself is a valid factor (which is handled in the next recursion level as new_remaining==1 case? No).
            # If new_remaining > 12 and new_remaining >= next_next_min, we can pick new_remaining as the next factor.
            # So the only dead end is if new_remaining < next_next_min (handled) or new_remaining <= 12 (invalid factor).
            # Wait, if new_remaining <= 12, it cannot be a factor.
            # So if new_remaining <= 12 and new_remaining > 1, it's a dead end.
            if new_remaining <= 12:
                continue
            
            # Recurse
            current_factors.append(f)
            dfs(new_remaining, current_factors, next_next_min, depth + 1)
            current_factors.pop()
            
            if time.time() > deadline:
                return
        
    dfs(target, [], 13, 0)
    
    return best
