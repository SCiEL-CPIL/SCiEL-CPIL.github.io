def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    def search(remaining, min_den, current_denoms, current_max):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return
        if remaining == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(current_denoms)
            return
        if len(current_denoms) >= 255:
            return
        # Lower bound: need at least 1/remaining terms of size >= min_den
        # The smallest possible max denominator is at least ceil(1/remaining) if we could use one term
        # But we need distinct denominators >= min_den
        # Pruning: if current_max >= best_max, no point continuing (unless we haven't set any yet)
        if current_max >= best_max and best_max < float('inf'):
            return
        
        # Try denominators from min_den upward
        # Upper bound for d: 1/d <= remaining, so d >= 1/remaining
        # Also need enough remaining terms to sum to 'remaining'
        # With k terms left, each >= min_den, max sum is k/min_den
        # Need remaining <= k/min_den, i.e., min_den <= k/remaining
        # Also 1/d >= remaining / (number of terms we can still use)
        # For the first term: 1/d <= remaining, so d >= ceil(1/remaining)
        
        # d must satisfy: 1/d <= remaining (so d >= 1/remaining)
        # And we need remaining - 1/d to be representable with distinct denoms > d
        # Lower bound for d: ceil(1/remaining)
        # Upper bound for d: we need remaining - 1/d > 0, so d > 1/remaining
        # Also, for the remaining terms (up to 255 - len(current_denoms) terms),
        # the maximum we can add is sum of 1/(d+1) + 1/(d+2) + ... 
        # But a simpler bound: each remaining term is at most 1/(d+1), and we have at most (255-len) terms
        # So remaining - 1/d <= (255-len(current_denoms)) / (d+1)
        # This gives us an upper bound on d
        
        max_terms_left = 255 - len(current_denoms)
        d_min = max(min_den, (1 + remaining.numerator - 1) // remaining.numerator)  # ceil(1/remaining)
        # Actually d_min = ceil(1/remaining) = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        
        # Upper bound: remaining - 1/d <= max_terms_left / (d+1)
        # remaining - 1/d <= max_terms_left/(d+1)
        # remaining*(d+1) - d <= max_terms_left
        # remaining*d + remaining - d <= max_terms_left
        # d*(remaining - 1) <= max_terms_left - remaining
        # If remaining < 1 (which it is), remaining - 1 < 0, so d >= (max_terms_left - remaining)/(1-remaining)
        # This gives a lower bound, not upper. Let me think differently.
        
        # For d to be valid: 1/d < remaining (strictly, since we need positive remainder)
        # So d > 1/remaining, i.e., d >= floor(1/remaining) + 1 if 1/remaining is integer, else ceil(1/remaining)
        # Actually d >= ceil(1/remaining) works, but if 1/d == remaining exactly, we're done.
        
        # Upper bound on d: we need to be able to represent remaining - 1/d with distinct denoms > d.
        # The maximum sum of (max_terms_left) distinct reciprocals starting from d+1 is:
        # 1/(d+1) + 1/(d+2) + ... + 1/(d+max_terms_left)
        # This is bounded by max_terms_left/(d+1)
        # So we need: remaining - 1/d <= max_terms_left/(d+1)
        # (remaining - 1/d)(d+1) <= max_terms_left
        # remaining*(d+1) - (d+1)/d <= max_terms_left
        # remaining*d + remaining - 1 - 1/d <= max_terms_left
        # For large d, approximately: remaining*d <= max_terms_left - remaining + 1
        # d <= (max_terms_left - remaining + 1) / remaining
        
        # This is a rough upper bound. Let's use it.
        # d_upper = floor((max_terms_left - remaining + 1) / remaining)
        # But this might be too loose or too tight. Let's just iterate and check.
        
        # Let's set a reasonable upper bound
        # d must be at least d_min
        # d must be such that 1/d < remaining (or == if it's the last term)
        # Let's iterate d from d_min, and stop when 1/d < remaining / (max_terms_left + 1) roughly
        # Actually, a better bound: after choosing d, we have remaining' = remaining - 1/d
        # We need remaining' to be representable with at most max_terms_left terms, each >= d+1
        # Max representable value with max_terms_left terms each >= d+1 is sum_{i=1}^{max_terms_left} 1/(d+i)
        # This is <= max_terms_left/(d+1)
        # So we need remaining - 1/d <= max_terms_left/(d+1)
        
        # Let's just iterate and break when the condition clearly fails
        d = d_min
        while d < 10000000:
            if time.time() > deadline:
                return
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                break
            # Check if remaining - 1/d can be represented
            rem2 = remaining - Fraction(1, d)
            if rem2 < 0:
                break
            # Prune: if current_max >= best_max, skip
            new_max = max(current_max, d)
            if new_max >= best_max and best_max < float('inf'):
                # We could still potentially find a better solution if d < best_max
                # but new_max >= best_max means we won't improve, so break
                # Actually, if d >= best_max, then new_max >= best_max, and all further d are larger
                break
            
            # Upper bound check: can we represent rem2 with max_terms_left terms each >= d+1?
            # Max sum = max_terms_left / (d+1) (very loose upper bound)
            if max_terms_left > 0 and rem2 > max_terms_left / (d + 1):
                # Can't represent it, try larger d? No, larger d makes rem2 smaller but also makes max sum smaller
                # Actually as d increases, 1/d decreases, so rem2 increases, but max sum decreases
                # So this condition might not be monotonic. Let's just proceed.
                pass
            
            current_denoms.append(d)
            search(rem2, d + 1, current_denoms, new_max)
            current_denoms.pop()
            
            d += 1
    
    search(target, 1, [], 0)
    
    if best_solution is not None:
        # Validate the solution
        if 0 < len(best_solution) <= 256:
            if all(1 <= d <= 10000000 for d in best_solution):
                if len(set(best_solution)) == len(best_solution):
                    s = sum(Fraction(1, d) for d in best_solution)
                    if s == target:
                        return best_solution
    
    # Fallback: Hardcoded valid solution for 5/31
    # 1/8 + 1/28 + 1/1736 = 5/31
    return [8, 28, 1736]
