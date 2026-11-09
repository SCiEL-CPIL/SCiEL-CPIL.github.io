def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    DEADLINE = 85.0
    
    target = Fraction(5, 31)
    best_result = None
    best_max_denom = float('inf')
    
    def search(remaining, used, current_list, max_allowed):
        nonlocal best_result, best_max_denom
        if time.time() > DEADLINE:
            return
        
        if remaining == 0:
            if current_list:
                max_d = max(current_list)
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_result = list(current_list)
            return
        
        if remaining.numerator <= 0:
            return
        
        if not current_list:
            min_d = 1
        else:
            min_d = max(current_list) + 1
        
        lower = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if lower <= min_d:
            lower = min_d + 1
        
        upper = min(max_allowed, lower + 200)
        
        for d in range(lower, upper + 1):
            if time.time() > DEADLINE:
                return
            if d in used:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                candidate = current_list + [d]
                max_d = max(candidate)
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_result = list(candidate)
                return
            search(new_remaining, used | {d}, current_list + [d], max_allowed)
    
    for max_allowed in range(7, 200):
        search(target, set(), [], max_allowed)
        if best_result is not None and best_max_denom <= max_allowed:
            break
    
    if best_result is None:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= 0 or d >= 10000000:
                break
            values.append(d)
            r -= Fraction(1, d)
        if values:
            return values
        return [31]
    
    return best_result
