def construct():
    from fractions import Fraction

    target = Fraction(5, 31)

    def find_sum(rem, max_den, max_terms, last_den):
        """
        Returns a list of denominators if 'rem' can be represented as sum of reciprocals 
        of 'max_terms' distinct integers, each <= max_den and > last_den.
        Returns None otherwise.
        """
        if rem == 0:
            return []
        if rem < 0:
            return None
        if max_terms == 0:
            return None
        
        # Lower bound for next denominator
        # 1/d <= rem => d >= 1/rem
        min_d = max(last_den + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if min_d > max_den:
            return None
        
        # Pruning: Check if max possible sum is sufficient
        # Max sum uses smallest available denominators: min_d, min_d+1, ...
        max_possible_sum = Fraction(0, 1)
        d = min_d
        for _ in range(max_terms):
            if d > max_den:
                break
            max_possible_sum += Fraction(1, d)
            d += 1
        if max_possible_sum < rem:
            return None
            
        # Iterate possible denominators for the next term
        for d in range(min_d, max_den + 1):
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
                
            sub_sol = find_sum(new_rem, max_den, max_terms - 1, d)
            if sub_sol is not None:
                return [d] + sub_sol
                
        return None

    # Search for the minimum largest denominator C
    # We assume the optimal solution has at most 6 terms total.
    # If it has fewer, find_sum will return early.
    
    for C in range(10, 1000):
        rem = target - Fraction(1, C)
        if rem <= 0:
            continue
            
        # Try to find 'rem' using up to 5 other denominators, all < C
        sol = find_sum(rem, C - 1, 5, 0)
        if sol is not None:
            final_sol = sol + [C]
            # Verification
            if sum(Fraction(1, x) for x in final_sol) == target:
                return final_sol
                
    # Fallback
    return [7, 8, 9, 10, 11]
