import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # Known solution: 1/7 + 1/56 + 1/1736 = 5/31 (max=1736)
    best_solution = [7, 56, 1736]
    best_max = 1736
    
    def check_valid(denoms):
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target
    
    def try_complete(remaining, min_d, max_allowed, current, max_terms=256):
        """Try to complete the Egyptian fraction using greedy + backtracking."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            if len(current) >= 1:
                return list(current)
            return None
        if remaining < 0:
            return None
        if len(current) >= max_terms:
            return None
        
        # Lower bound on next denominator
        d_lo = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if d_lo < min_d:
            d_lo = min_d
        if d_lo > max_allowed:
            return None
        
        # Try denominators from d_lo up to max_allowed
        # Use a limited search with pruning
        max_search = min(max_allowed, d_lo + 200)
        
        for d in range(d_lo, max_search + 1):
            if time.time() > deadline:
                return None
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                result = current + [d]
                if check_valid(result):
                    return result
                continue
            # Prune: check if remaining is achievable with remaining terms
            # Each future term <= 1/(d+1), so new_remaining <= (max_terms - len(current) - 1) / (d+1)
            terms_left = max_terms - len(current) - 1
            if terms_left > 0 and new_remaining > Fraction(terms_left, d + 1):
                continue
            # Also: new_remaining >= 1/(max_allowed) * 1 (at least one more term needed)
            if terms_left > 0 and new_remaining < Fraction(1, max_allowed):
                # Can't be achieved with terms bounded by max_allowed
                # Actually this is wrong - we need new_remaining to be sum of terms each >= 1/max_allowed
                # So new_remaining >= 1/max_allowed if we need at least 1 more term
                # But we might need more terms, so this bound is loose
                pass
            
            result = try_complete(new_remaining, d + 1, max_allowed, current + [d], max_terms)
            if result is not None:
                return result
        
        return None
    
    def search_with_max_denom(K, max_terms=256):
        """Search for a solution with all denominators <= K."""
        if time.time() > deadline:
            return None
        
        # Try all possible first denominators
        # 1/a <= 5/31 => a >= 7
        # Also 1/a >= 5/31 - (max_terms-1)/K approximately
        a_min = 7
        a_max = min(K, 100)  # Limit first denominator search
        
        for a in range(a_min, a_max + 1):
            if time.time() > deadline:
                return None
            ra = target - Fraction(1, a)
            if ra <= 0:
                continue
            if ra.numerator == 1:
                # ra = 1/b, check if b > a and b <= K
                b = ra.denominator
                if b > a and b <= K:
                    sol = [a, b]
                    if check_valid(sol):
                        return sol
                continue
            
            # Try all possible second denominators
            b_lo = (ra.denominator + ra.numerator - 1) // ra.numerator
            if b_lo <= a:
                b_lo = a + 1
            b_hi = min(K, b_lo + 100)  # Limit second denominator search
            
            for b in range(b_lo, b_hi + 1):
                if time.time() > deadline:
                    return None
                rb = ra - Fraction(1, b)
                if rb <= 0:
                    break
                if rb.numerator == 1:
                    c = rb.denominator
                    if c > b and c <= K:
                        sol = [a, b, c]
                        if check_valid(sol):
                            return sol
                    continue
                
                # Try to complete with more terms
                result = try_complete(rb, b + 1, K, [a, b], max_terms)
                if result is not None:
                    return result
        
        return None
    
    # Binary search on K
    # Lower bound: at least 7 (since 1/7 < 5/31)
    # Upper bound: 1736 (known solution)
    lo = 7
    hi = 1736
    
    # First, try to find if there's a solution with K < 1736
    # Use binary search
    found = False
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        result = search_with_max_denom(mid)
        if result is not None:
            mx = max(result)
            if mx < best_max:
                best_max = mx
                best_solution = result
            hi = mid - 1
            found = True
        else:
            lo = mid + 1
    
    # If binary search didn't find better, try a more exhaustive search
    if best_max == 1736 and time.time() < deadline - 5:
        # Try K from 1735 down to 1000 in steps
        for K in range(1735, 999, -5):
            if time.time() > deadline - 5:
                break
            result = search_with_max_denom(K)
            if result is not None:
                mx = max(result)
                if mx < best_max:
                    best_max = mx
                    best_solution = result
                break
    
    # Final verification
    if not check_valid(best_solution):
        best_solution = [7, 56, 1736]
        best_max = 1736
    
    return best_solution
