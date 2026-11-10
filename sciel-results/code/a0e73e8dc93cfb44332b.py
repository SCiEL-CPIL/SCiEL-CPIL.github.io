from math import gcd
from fractions import Fraction

def construct():
    # Find minimum number of distinct denominators >= 5 summing to 1
    # Try n = 5, 6, 7, 8, ... until we find a solution
    
    for n in range(5, 13):
        result = _search(n)
        if result is not None:
            return result
    raise RuntimeError("no solution found")

def _search(n):
    # Find n distinct denominators >= 5 with sum of reciprocals = 1
    # Use backtracking
    
    target = Fraction(1, 1)
    
    def backtrack(start, remaining, chosen, remaining_terms):
        if remaining_terms == 0:
            return chosen if remaining == 0 else None
        
        if remaining <= 0:
            return None
        
        # Maximum possible sum with remaining_terms terms starting from 'start'
        # The largest reciprocals we can use are 1/start, 1/(start+1), ...
        # But we need to check if we can reach 'remaining'
        
        # Lower bound: use the smallest possible denominators (largest reciprocals)
        # to see if we can reach 'remaining'
        # Upper bound: use the largest possible denominators (smallest reciprocals)
        
        # Pruning 1: even with the largest possible reciprocals (smallest denominators),
        # we can't reach 'remaining'
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible += Fraction(1, start + i)
        if max_possible < remaining:
            return None
        
        # Pruning 2: even with the smallest possible reciprocals (largest denominators),
        # we exceed 'remaining'. But denominators are unbounded above, so this is tricky.
        # Instead, if 1/start >= remaining and remaining_terms > 1, we might still work.
        # Let's just try each candidate.
        
        # The next denominator must be at least 'start'
        # And 1/d < remaining (since we need more terms)
        # So d > 1/remaining
        # Also, we need remaining_terms-1 more terms, each with denominator > d
        # So the max sum of remaining is sum of 1/(d+1) + 1/(d+2) + ... 
        # Let's iterate d from start upward
        
        # Upper bound for d: 1/d < remaining => d > 1/remaining
        # But also, we need to leave enough for the other terms.
        # The minimum sum of the other (remaining_terms-1) terms is achieved with very large denominators,
        # approaching 0. So we just need 1/d < remaining.
        # But also, the maximum sum of the other terms is bounded.
        
        # Let's set an upper bound: d must satisfy that even with the best case for remaining terms,
        # we don't overshoot. Actually, let's just iterate and prune.
        
        # d must be such that 1/d <= remaining (otherwise we overshoot)
        # So d >= ceil(1/remaining) if remaining > 0
        # But d >= start as well
        
        # Also, we need to be able to represent (remaining - 1/d) with (remaining_terms-1) terms
        # each with denominator > d
        
        # Upper bound for d: since we need (remaining_terms-1) more terms each with denominator > d,
        # the maximum they can contribute is sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # So we need remaining - 1/d <= sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # This gives a lower bound on how small d can be... actually it's complex.
        
        # Let's just iterate d from start, with a reasonable upper bound.
        # Upper bound: d such that 1/d < remaining, so d > 1/remaining.
        # But also, d can't be too large because we need to make up the difference.
        # A safe upper bound: d <= some limit. Since remaining_terms is small, 
        # let's use d up to maybe 1000000 but with early termination.
        
        # Actually, let's use a smarter bound. The next term 1/d must be < remaining.
        # And the remaining (remaining_terms-1) terms must sum to (remaining - 1/d).
        # Each of those terms is at most 1/(d+1), so:
        # remaining - 1/d <= (remaining_terms-1) / (d+1)
        # This gives us a constraint on d.
        
        # Let's solve: remaining - 1/d <= (remaining_terms-1)/(d+1)
        # (remaining*d - 1)/d <= (remaining_terms-1)/(d+1)
        # (remaining*d - 1)*(d+1) <= d*(remaining_terms-1)
        # This is a quadratic, let's just compute it numerically for each d.
        
        for d in range(start, 10000000):
            if Fraction(1, d) >= remaining:
                # 1/d >= remaining, so either it's exactly equal (if remaining_terms==1) or too big
                if remaining_terms == 1:
                    if Fraction(1, d) == remaining:
                        return chosen + [d]
                # If 1/d > remaining, and d increases, 1/d decreases, so continue
                # But if 1/d == remaining and remaining_terms > 1, we can't use more terms
                if Fraction(1, d) > remaining:
                    continue
                else:
                    # 1/d == remaining but we need more terms, so skip
                    continue
            
            # Check if we can complete with remaining_terms-1 terms
            new_remaining = remaining - Fraction(1, d)
            if new_remaining <= 0:
                continue
            
            # Check upper bound: can (remaining_terms-1) terms with denominators > d sum to new_remaining?
            # Max sum is sum_{i=1}^{remaining_terms-1} 1/(d+i)
            max_sum = Fraction(0, 1)
            for i in range(1, remaining_terms):
                max_sum += Fraction(1, d + i)
            if max_sum < new_remaining:
                # Even with the best case, we can't reach new_remaining
                # As d increases, max_sum decreases, so we can break
                break
            
            # Also check: min sum with (remaining_terms-1) terms is approaching 0,
            # so no lower bound issue.
            
            # Recurse
            result = backtrack(d + 1, new_remaining, chosen + [d], remaining_terms - 1)
            if result is not None:
                return result
        
        return None
    
    # Start with smallest denominator 5
    return backtrack(5, target, [], n)
