"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    import math
    
    target = Fraction(5, 31)
    start_time = time.time()
    timeout = 55
    
    best_max = [10000000]
    best_solution = [[]]
    
    def search(remaining, min_den, current_terms, max_allowed):
        if time.time() - start_time > timeout:
            return
        
        if remaining == 0:
            if current_terms:
                max_den = max(current_terms)
                if max_den < best_max[0]:
                    best_max[0] = max_den
                    best_solution[0] = current_terms[:]
            return
        
        if remaining < 0:
            return
        
        if remaining > 0:
            min_next = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            min_next = max(min_next, min_den)
            
            if min_next > max_allowed:
                return
            
            # Pruning: approximate harmonic sum from min_next to max_allowed
            if min_next > 1:
                approx_sum = math.log(max_allowed / min_next) + 1.0 / min_next
            else:
                approx_sum = math.log(max_allowed) + 1.0
            
            if approx_sum < float(remaining) * 0.9:
                return
            
            for d in range(min_next, max_allowed + 1):
                if time.time() - start_time > timeout:
                    return
                
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    continue
                
                current_terms.append(d)
                if new_remaining == 0:
                    if current_terms:
                        max_den = max(current_terms)
                        if max_den < best_max[0]:
                            best_max[0] = max_den
                            best_solution[0] = current_terms[:]
                else:
                    search(new_remaining, d + 1, current_terms, max_allowed)
                current_terms.pop()
    
    # Get an initial upper bound using greedy
    r = target
    initial_terms = []
    min_d = 1
    while r > 0 and time.time() - start_time < timeout:
        d = (r.denominator + r.numerator - 1) // r.numerator
        d = max(d, min_d)
        initial_terms.append(d)
        r -= Fraction(1, d)
        min_d = d + 1
    
    if initial_terms:
        best_max[0] = max(initial_terms)
        best_solution[0] = initial_terms[:]
    
    # Try to find a better solution
    upper = best_max[0]
    
    # Test various max_allowed values
    test_values = list(range(15, min(upper + 1, 100)))
    if upper > 100:
        test_values.extend(range(100, min(upper + 1, 500), 10))
        test_values.extend(range(500, upper + 1, 50))
    
    for max_allowed in test_values:
        if max_allowed >= best_max[0]:
            break
        if time.time() - start_time > timeout:
            break
        search(target, 1, [], max_allowed)
    
    if not best_solution[0]:
        return [8, 28, 1736]
    return best_solution[0]

# EVOLVE-BLOCK-END
