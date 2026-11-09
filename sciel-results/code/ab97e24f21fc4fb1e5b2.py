def construct():
    from fractions import Fraction
    import time
    
    target = Fraction(5, 31)
    max_terms = 256
    max_den_global = 10000000
    start_time = time.time()
    time_limit = 55.0
    
    def max_sum_with_k_terms(min_d, k, max_d):
        if k <= 0 or min_d > max_d:
            return Fraction(0, 1)
        total = Fraction(0, 1)
        for i in range(k):
            d = min_d + i
            if d > max_d:
                break
            total += Fraction(1, d)
        return total
    
    def min_sum_with_k_terms(min_d, k, max_d):
        if k <= 0 or min_d > max_d:
            return Fraction(0, 1)
        total = Fraction(0, 1)
        for i in range(k):
            d = max_d - i
            if d < min_d:
                break
            total += Fraction(1, d)
        return total
    
    def dfs(remaining, min_d, max_d, depth, current_terms):
        if time.time() - start_time > time_limit:
            return False
        
        if remaining == 0:
            return len(current_terms) > 0
        
        if depth >= max_terms or remaining < 0:
            return False
        
        k = max_terms - depth
        
        max_possible = max_sum_with_k_terms(min_d, k, max_d)
        if remaining > max_possible:
            return False
        
        min_possible = min_sum_with_k_terms(min_d, k, max_d)
        if remaining < min_possible:
            return False
        
        if remaining > 0:
            min_d_candidate = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        else:
            return False
        
        for d in range(min_d_candidate, max_d + 1):
            if time.time() - start_time > time_limit:
                return False
            
            if Fraction(1, d) > remaining:
                break
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            
            current_terms.append(d)
            if dfs(new_remaining, d + 1, max_d, depth + 1, current_terms):
                return True
            current_terms.pop()
        
        return False
    
    # Get an upper bound from greedy
    r = target
    greedy_vals = []
    while r and len(greedy_vals) < max_terms:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d > max_den_global:
            break
        greedy_vals.append(d)
        r -= Fraction(1, d)
    
    hi = max(greedy_vals) if greedy_vals else max_den_global
    lo = 1
    result = None
    
    while lo <= hi and time.time() - start_time < time_limit:
        mid = (lo + hi) // 2
        current_terms = []
        if dfs(target, 1, mid, 0, current_terms):
            result = current_terms
            hi = mid - 1
        else:
            lo = mid + 1
    
    if result:
        return result
    
    return greedy_vals if greedy_vals else [8]
