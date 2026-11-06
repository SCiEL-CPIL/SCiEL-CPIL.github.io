"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """
    Find the minimum number of distinct integer denominators in [5, 10000000]
    whose reciprocals sum to exactly 1.
    
    We know from analysis that 3 terms is impossible with denominators >= 5.
    So we try 4, then 5, etc.
    
    Strategy: Backtracking search with strong pruning.
    """
    target = Fraction(1, 1)
    
    def search(remaining, min_denom, terms_left, path):
        """
        Find a representation of 'remaining' as a sum of exactly 'terms_left'
        unit fractions with distinct denominators >= min_denom.
        Returns list of denominators or None.
        """
        if terms_left == 0:
            return list(path) if remaining == 0 else None
        
        if remaining == 0:
            # Can't add more terms if remaining is 0
            return None
        
        # Pruning: with terms_left unit fractions, each at most 1/min_denom,
        # the maximum possible sum is terms_left / min_denom
        if remaining > terms_left * Fraction(1, min_denom):
            return None
        
        # Pruning: the minimum possible sum with terms_left unit fractions
        # with distinct denominators starting from min_denom is
        # 1/min_denom + 1/(min_denom+1) + ...
        # This is a lower bound check - if even the smallest possible sum
        # exceeds remaining, we can't use this path. Actually, we need
        # the SUM to equal remaining, and each term is positive, so
        # remaining must be at least the sum of the terms_left smallest
        # possible unit fractions starting from min_denom.
        # But actually we just need remaining to be achievable, and since
        # we can always choose larger denominators, the constraint is
        # remaining <= terms_left / min_denom (already checked above).
        # Also remaining >= sum of terms_left smallest unit fractions? No,
        # we want the sum to EQUAL remaining, so we need remaining to be
        # large enough to be the sum of terms_left distinct unit fractions
        # each >= 1/(some large denom). The minimum sum with terms_left 
        # terms is when denominators are as large as possible, which approaches 0.
        # The maximum sum is when denominators are smallest: 1/min_denom + 1/(min_denom+1) + ...
        # So we need: sum_{k=0}^{terms_left-1} 1/(min_denom+k) >= remaining
        min_possible = sum(Fraction(1, min_denom + k) for k in range(terms_left))
        if remaining > min_possible:
            return None
        
        # For the last term: 1/d = remaining, so d = 1/remaining must be integer
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_denom and d <= 10000000:
                    return list(path) + [d]
            return None
        
        # Determine range of valid denominators for this term
        # 1/d <= remaining => d >= 1/remaining
        # 1/d >= remaining / terms_left => d <= terms_left / remaining
        # Also d >= min_denom
        
        min_d = max(min_denom, remaining.denominator // remaining.numerator + (1 if remaining.denominator % remaining.numerator else 0))
        # More precisely: d >= ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
        min_d = max(min_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        max_d = min(10000000, terms_left * remaining.denominator // remaining.numerator + 1)
        
        if min_d > max_d:
            return None
        
        # Try denominators from min_d to max_d
        # For efficiency, try a limited set first
        # We can also use the fact that for the remaining terms_left - 1 terms,
        # we need remaining - 1/d to be representable.
        
        for d in range(min_d, max_d + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                if terms_left == 1:
                    return list(path) + [d]
                else:
                    continue  # Can't have remaining terms sum to 0
            if terms_left - 1 >= 1:
                # Check if new_remaining can be represented with terms_left-1 terms
                # starting from d+1
                result = search(new_remaining, d + 1, terms_left - 1, path + [d])
                if result is not None:
                    return result
        
        return None
    
    # Try with increasing number of terms starting from 3
    for num_terms in range(3, 20):
        result = search(target, 5, num_terms, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
