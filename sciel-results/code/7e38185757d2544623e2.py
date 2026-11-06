"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    target = Fraction(1, 1)
    
    # Determine the minimum number of terms needed based on the maximum possible sum
    # with k distinct denominators starting from 5.
    min_terms_needed = 1
    current_sum = 0
    for k in range(1, 100):
        current_sum += 1 / (4 + k)
        if current_sum >= 1:
            min_terms_needed = k
            break

    def dfs(remaining, min_denom, terms_left, path):
        if terms_left == 0:
            return None if remaining != 0 else list(path)
        
        if remaining == 0:
            return list(path)
            
        # Pruning: Check if the maximum possible sum with 'terms_left' terms
        # starting from 'min_denom' is at least 'remaining'.
        # The max sum is achieved by taking the smallest available denominators.
        if terms_left <= 50:
            max_possible = sum(Fraction(1, i) for i in range(min_denom, min_denom + terms_left))
            if max_possible < remaining:
                return None
        else:
            # Approximation for large terms_left
            import math
            max_possible_approx = math.log(min_denom + terms_left) - math.log(min_denom)
            if max_possible_approx < float(remaining):
                return None

        # Bounds for the next denominator d
        # 1/d <= remaining  =>  d >= 1/remaining
        # 1/d >= remaining / terms_left  =>  d <= terms_left / remaining
        lower_d = max(min_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        upper_d = terms_left * remaining.denominator // remaining.numerator
        
        if lower_d > upper_d:
            return None
            
        # Optimization for the last term
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_denom and d <= upper_d:
                    return list(path) + [d]
            return None

        # Try denominators from smallest to largest (greedy)
        for d in range(lower_d, upper_d + 1):
            next_remaining = remaining - Fraction(1, d)
            if next_remaining < 0:
                break
            if next_remaining == 0:
                # We have satisfied the sum, but we have terms_left - 1 terms left.
                # This is a valid solution of length len(path) + 1.
                # Since we search for increasing k, this will be the first time we find
                # a solution of this length if no shorter solution exists.
                return list(path) + [d]
            
            # Recurse
            result = dfs(next_remaining, d + 1, terms_left - 1, path + [d])
            if result is not None:
                return result
                
        return None

    # Search for the minimum number of terms
    for k in range(min_terms_needed, 30):
        result = dfs(target, 5, k, [])
        if result is not None:
            # Final verification
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
                
    raise RuntimeError("No representation found")
# EVOLVE-BLOCK-END
