from fractions import Fraction

def construct():
    # We need to find 1..256 distinct integer denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1, minimizing the count.
    
    # Let's start with a known small solution and try to optimize it.
    # A classic Egyptian fraction for 1 is:
    # 1 = 1/2 + 1/3 + 1/6
    # But we need denominators >= 5.
    # 
    # Let's think about this differently. We want to represent 1 as a sum of
    # distinct unit fractions with denominators >= 5, using as few terms as possible.
    #
    # With 1 term: max is 1/5 < 1, impossible.
    # With 2 terms: max is 1/5 + 1/6 = 11/30 < 1, impossible.
    # With 3 terms: max is 1/5 + 1/6 + 1/7 = (42+35+30)/210 = 107/210 < 1, impossible.
    # With 4 terms: max is 1/5 + 1/6 + 1/7 + 1/8 = (168+140+120+105)/840 = 533/840 < 1, impossible.
    # With 5 terms: max is 1/5+1/6+1/7+1/8+1/9 = (504+420+360+315+280)/2520 = 1879/2520 < 1, impossible.
    # With 6 terms: 1/5+1/6+1/7+1/8+1/9+1/10 = 1879/2520 + 1/10 = 1879/2520 + 252/2520 = 2131/2520 < 1, impossible.
    # With 7 terms: add 1/11: 2131/2520 + 1/11 = (2131*11 + 2520)/(2520*11) = (23441+2520)/27720 = 25961/27720 < 1.
    # With 8 terms: add 1/12: 25961/27720 + 1/12 = (25961*12 + 27720)/(27720*12) = (311532+27720)/332640 = 339252/332640 > 1.
    # So 8 terms is possible in principle.
    
    # Let's use a greedy-like approach with backtracking for small n.
    # Start with n=8 and search for solutions.
    
    for n in range(8, 257):
        result = _search(n)
        if result is not None:
            return result
    
    raise RuntimeError("no representation found")

def _search(n):
    """Find n distinct denominators >= 5 with reciprocal sum = 1 using backtracking."""
    target = Fraction(1, 1)
    
    def backtrack(remaining, min_denom, terms_left, chosen):
        if terms_left == 0:
            return chosen if remaining == 0 else None
        
        if remaining <= 0:
            return None
        
        # Pruning: maximum possible sum with terms_left terms starting from min_denom
        # This is sum(1/i for i in range(min_denom, min_denom + terms_left))
        # But computing this exactly with Fractions each time is expensive.
        # For large min_denom, this sum is small.
        
        # The next denominator d must satisfy:
        # 1. d >= min_denom
        # 2. 1/d <= remaining  =>  d >= ceil(1/remaining)
        # 3. remaining - 1/d must be achievable with terms_left-1 terms all > d
        
        # Lower bound for d
        if remaining < 1:
            # 1/d <= remaining => d >= 1/remaining
            min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        else:
            min_d = min_denom
        
        min_d = max(min_d, min_denom)
        
        # Upper bound for d: we need to be able to reach remaining with terms_left terms
        # starting from min_d. The maximum sum is achieved by taking the smallest denominators.
        # Also, d cannot be so large that even with terms_left-1 more terms we can't reach remaining.
        # If 1/d + sum of largest (terms_left-1) unit fractions after d < remaining, then d is too large.
        # But the largest unit fractions after d are 1/(d+1), 1/(d+2), ...
        # So we need 1/d + 1/(d+1) + ... + 1/(d+terms_left-1) >= remaining
        
        # Let's iterate d from min_d upward, with a reasonable cap.
        # The cap should be such that 1/d + (terms_left-1) * 1/(d+1) >= remaining (rough upper bound)
        # Or more simply, just try a bounded range.
        
        # For efficiency, let's limit the search space.
        # If d is very large, 1/d is very small, and we'd need many small terms to make up the difference.
        
        max_search = min_d + 2000  # Reasonable cap for each level
        
        for d in range(min_d, min(max_search, 10000000) + 1):
            if d > 10000000:
                break
            
            frac = Fraction(1, d)
            if frac > remaining:
                continue
            
            new_remaining = remaining - frac
            if new_remaining < 0:
                break
            
            result = backtrack(new_remaining, d + 1, terms_left - 1, chosen + [d])
            if result is not None:
                return result
        
        return None
    
    return backtrack(target, 5, n, [])
