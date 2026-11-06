def construct():
    import time
    from fractions import Fraction
    
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def find_solution(num, den, terms_left, used_set, max_denom):
        """
        Find distinct denominators d1, ..., dk (k = terms_left) such that
        1/d1 + ... + 1/dk = num/den, all di <= max_denom, di not in used_set.
        Returns list of denominators or None.
        """
        if terms_left == 0:
            return [] if num == 0 else None
        
        if num <= 0:
            return None
        
        if terms_left == 1:
            # 1/d = num/den => d = den/num
            if den % num != 0:
                return None
            d = den // num
            if d < 1 or d > max_denom or d in used_set:
                return None
            return [d]
        
        # min_d: 1/d <= num/den => d >= den/num
        min_d = (den + num - 1) // num
        if min_d < 1:
            min_d = 1
        
        # For the remaining terms, we need them to be able to sum to the new remainder
        # After picking d, new remainder = (num*d - den)/(den*d)
        # This must be positive: num*d > den, already ensured by min_d
        
        max_d = max_denom
        
        for d in range(min_d, max_d + 1):
            if d in used_set:
                continue
            
            new_num = num * d - den
            if new_num <= 0:
                continue
            
            new_den = den * d
            
            g = _gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            new_used = used_set | {d}
            
            result = find_solution(new_num, new_den, terms_left - 1, new_used, max_denom)
            if result is not None:
                return [d] + result
        
        return None
    
    # Get initial greedy solution
    def greedy_solve():
        r = target
        denoms = []
        used = set()
        while r > 0:
            k = (r.denominator + r.numerator - 1) // r.numerator
            while k in used:
                k += 1
            denoms.append(k)
            used.add(k)
            r -= Fraction(1, k)
        return denoms
    
    initial_greedy = greedy_solve()
    best_solution = initial_greedy
    best_max = max(initial_greedy)
    
    # Try different numbers of terms, from small to larger
    # For each term count, binary search on max denominator
    for num_terms in range(3, min(len(initial_greedy) + 3, 15)):
        if time.time() > deadline - 5:
            break
        
        # Lower bound on max denom: at least 7
        # Upper bound: best_max
        lo = 7
        hi = best_max
        
        if lo >= hi:
            continue
        
        while lo < hi:
            if time.time() > deadline - 5:
                break
            
            mid = (lo + hi) // 2
            
            sol = find_solution(target.numerator, target.denominator, num_terms, set(), mid)
            
            if sol is not None and max(sol) <= mid:
                current_max = max(sol)
                if current_max < best_max:
                    best_max = current_max
                    best_solution = sol
                hi = mid
            else:
                lo = mid + 1
        
        # After binary search, try lo (the smallest feasible)
        if time.time() < deadline - 5:
            sol = find_solution(target.numerator, target.denominator, num_terms, set(), lo)
            if sol is not None and max(sol) < best_max:
                best_max = max(sol)
                best_solution = sol
    
    return best_solution
