import math
import time
from fractions import Fraction

# Known solutions for different k values
KNOWN_SOLUTIONS = {
    7: [5, 6, 7, 42, 140, 210, 420],  # Verify: 1/5+1/6+1/7+1/42+1/140+1/210+1/420
    8: [5, 6, 7, 8, 9, 10, 14, 252],  # Verify: 1/5+1/6+1/7+1/8+1/9+1/10+1/14+1/252
    9: [5, 6, 7, 8, 9, 10, 14, 15, 63],  # Verify: 1/5+1/6+1/7+1/8+1/9+1/10+1/14+1/15+1/63
    10: [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
}

def verify_solution(denoms):
    """Verify that the sum of reciprocals equals exactly 1."""
    s = Fraction(0, 1)
    for d in denoms:
        s += Fraction(1, d)
    return s == 1

def find_solution_backtrack(k, deadline):
    """
    Find k distinct denominators >= 5 summing to 1.
    Returns list of denominators or None.
    """
    target = Fraction(1, 1)
    
    def max_sum_from(start_d, terms):
        """Calculate max sum of 'terms' distinct reciprocals starting from start_d."""
        s = Fraction(0, 1)
        for i in range(terms):
            s += Fraction(1, start_d + i)
        return s

    def dfs(start_d, remaining_sum, terms_left, current_solution):
        if time.time() > deadline:
            return None
            
        if terms_left == 0:
            if remaining_sum == 0:
                return current_solution
            return None
            
        if terms_left == 1:
            # 1/d = remaining_sum => d = 1/remaining_sum
            if remaining_sum == 0:
                return None
            d = int(1 / remaining_sum)
            # Check if it's an integer and >= start_d
            if remaining_sum * d == 1 and d >= start_d:
                return current_solution + [d]
            return None

        # Lower bound for d: 1/d <= remaining_sum => d >= 1/remaining_sum
        # Since d is integer, d >= ceil(1/remaining_sum)
        # Also d >= start_d
        min_d = max(start_d, math.ceil(1 / float(remaining_sum)))
        
        # Upper bound for d:
        # We need sum of 'terms_left' reciprocals starting from d+1 to be >= remaining_sum
        # Approx: terms_left / (d+1) >= remaining_sum
        # d+1 <= terms_left / remaining_sum
        # d <= terms_left / remaining_sum - 1
        # Using float for bound estimation is safe for pruning, verify with exact logic
        max_d_float = float(terms_left) / float(remaining_sum) - 1
        max_d = int(max_d_float)
        
        # Also, if we pick d, the next start is d+1.
        # The max sum we can achieve with terms_left-1 terms starting from d+1 is max_sum_from(d+1, terms_left-1).
        # This must be >= remaining_sum - 1/d.
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
                
            next_remaining = remaining_sum - Fraction(1, d)
            if next_remaining < 0:
                break # Since d is increasing, 1/d is decreasing, so next_remaining will only increase? 
                # Wait. d increases => 1/d decreases => next_remaining increases.
                # So if next_remaining < 0, it means 1/d > remaining_sum.
                # This happens if d < 1/remaining_sum. But we started from ceil(1/remaining_sum).
                # So this shouldn't happen unless float error in min_d.
                # Actually if d is too small, 1/d is too big.
                # So if next_remaining < 0, we should continue to larger d?
                # No, if d is small, 1/d is large. If 1/d > remaining_sum, we overshoot.
                # We need 1/d <= remaining_sum.
                # So d >= 1/remaining_sum.
                # My min_d logic handles this.
                
            # Check if it's possible to reach next_remaining with terms_left-1 terms starting from d+1
            # Max sum of terms_left-1 terms starting from d+1
            if terms_left - 1 > 0:
                # Optimization: calculate max sum efficiently or use approximation
                # For small terms_left (<= 7), simple sum is fast
                max_possible = Fraction(0, 1)
                for i in range(terms_left - 1):
                    max_possible += Fraction(1, d + 1 + i)
                
                if max_possible < next_remaining:
                    # Even with smallest possible denominators, we can't reach the sum
                    # Increasing d will make max_possible even smaller
                    break
            
            # Recurse
            res = dfs(d + 1, next_remaining, terms_left - 1, current_solution + [d])
            if res is not None:
                return res
                
        return None

    # Start search
    # First denominator d1.
    # Max sum of k-1 terms starting from d1+1 must be >= 1 - 1/d1
    # Min d1 is 5.
    # Max d1: 1/d1 <= 1 => d1 >= 1.
    # Also 1/d1 + max_sum(1, k-1, d1+1) >= 1
    
    for d1 in range(5, 20): # d1 won't be very large for small k
        if time.time() > deadline:
            return None
        next_remaining = 1 - Fraction(1, d1)
        if next_remaining <= 0:
            continue
        
        # Check max possible sum for remaining k-1 terms
        max_possible = Fraction(0, 1)
        for i in range(k - 1):
            max_possible += Fraction(1, d1 + 1 + i)
            
        if max_possible < next_remaining:
            continue
            
        res = dfs(d1 + 1, next_remaining, k - 1, [d1])
        if res is not None:
            return res
            
    return None

def construct():
    start_time = time.time()
    # 90s limit. Use 80s for safety.
    deadline = start_time + 80.0
    
    # First, try known solutions from smallest k
    for k in sorted(KNOWN_SOLUTIONS.keys()):
        sol = KNOWN_SOLUTIONS[k]
        if verify_solution(sol):
            return sol
    
    # If no known solution works, use backtracking search
    # Try k=7 first (smallest possible)
    for k in [7, 8, 9, 10]:
        if time.time() > deadline - 5.0:
            break
        sol = find_solution_backtrack(k, deadline)
        if sol is not None:
            return sol
    
    # Fallback to a verified solution
    return [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
