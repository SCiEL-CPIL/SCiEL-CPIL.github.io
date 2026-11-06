from fractions import Fraction
import time

def construct():
    deadline = time.time() + 85
    
    # Check N from 3 to 20
    for N in range(3, 21):
        if time.time() > deadline:
            break
            
        # Search for solution with N terms
        # State: remaining_sum, min_denom, used_set
        # We want to find denominators d1 < d2 < ... < dN
        # such that sum(1/di) = 1
        
        # Heuristic: Start with smallest possible denominators to keep sum high, 
        # then adjust.
        
        # Recursive backtracking
        # To speed up, we can precompute or estimate bounds.
        
        def max_sum(start, k):
            """Max sum of k distinct unit fractions with denominators >= start"""
            s = Fraction(0, 1)
            for i in range(k):
                s += Fraction(1, start + i)
            return s

        def search(remaining, min_d, count, path):
            if time.time() > deadline:
                return None
                
            if count == 0:
                if remaining == 0:
                    return path[:]
                return None
            
            if count == 1:
                # remaining must be 1/d
                if remaining.numerator == 1:
                    d = remaining.denominator
                    if d >= min_d and d not in path:
                        return path + [d]
                return None
            
            # Lower bound for d: 1/d <= remaining => d >= 1/remaining
            # Also d >= min_d
            low_d = max(min_d, (1 / remaining).numerator if remaining.denominator == 1 else (1 / remaining).numerator // remaining.denominator + 1)
            # Actually 1/remaining is a float. Let's use integer division.
            # d >= ceil(1/remaining)
            if remaining > 0:
                low_d = max(min_d, (1 / remaining).numerator if remaining.denominator == 1 else (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            else:
                return None
            
            # Upper bound for d:
            # We need remaining - 1/d <= max_sum(d+1, count-1)
            # Also 1/d < remaining => d > 1/remaining
            
            # Iterate d
            # To limit search space, we might need a good upper bound for d.
            # If d is too large, 1/d is too small, and we might not be able to sum up to 'remaining' 
            # with the remaining terms? No, we can always make terms smaller by increasing denominators.
            # Wait, we want to EXACTLY match 'remaining'.
            # If we pick a very large d, 1/d is very small. The remaining sum is close to 'remaining'.
            # We need to form 'remaining - 1/d' with count-1 terms.
            # The MAX sum of count-1 terms starting from d+1 is max_sum(d+1, count-1).
            # If remaining - 1/d > max_sum(d+1, count-1), it's impossible.
            # So we need 1/d >= remaining - max_sum(d+1, count-1).
            
            # Let's iterate d from low_d upwards.
            # We should stop if 1/d < remaining - max_sum(d+1, count-1) is no longer satisfiable?
            # Actually, as d increases, 1/d decreases.
            # remaining - 1/d increases.
            # max_sum(d+1, count-1) decreases.
            # So the condition (remaining - 1/d) <= max_sum(d+1, count-1) will eventually fail.
            # So there is an upper bound on d.
            
            # To avoid infinite loop or too large d, we can set a soft limit or break when condition fails.
            # Condition: remaining - 1/d <= max_sum(d+1, count-1)
            
            d = low_d
            # We need to be careful with the loop size.
            # For N=8, the search space is manageable.
            
            while True:
                if time.time() > deadline:
                    return None
                
                if d > 10000000: # Denominator cap
                    break
                
                # Check if 1/d <= remaining
                if Fraction(1, d) > remaining:
                    d += 1
                    continue
                
                rest = remaining - Fraction(1, d)
                
                # Check if rest can be formed by count-1 terms starting from d+1
                if rest > max_sum(d + 1, count - 1):
                    # rest is too large, we need smaller d (larger 1/d)
                    # But we are iterating d upwards. So we should have checked smaller d first.
                    # If we are here, it means for this d, the max possible sum of the rest is too small.
                    # As d increases, 1/d decreases, so rest increases.
                    # And max_sum(d+1, count-1) decreases.
                    # So the gap widens. We can break.
                    break
                
                # Check if rest > 0
                if rest < 0:
                    d += 1
                    continue
                
                # Recursive call
                res = search(rest, d + 1, count - 1, path + [d])
                if res is not None:
                    return res
                
                d += 1
            
            return None
        
        sol = search(Fraction(1, 1), 5, N, [])
        if sol is not None:
            return sol
            
    # Fallback if not found (should not happen for N up to 20)
    # Return a valid solution, maybe not minimal
    # 1/5 + 1/6 + 1/30 = 2/5.
    # 2/5 = 1/5 + 1/10 + 1/20 ...
    # Just return a known valid set, e.g. from current program logic or a hardcoded one.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # Sum is approx 1.5. Too high.
    
    # We can use the current program's logic as a fallback, but it's slow.
    # Let's try to find a solution with N=10 or so quickly if the above fails.
    # But the above should find N=8.
    
    raise RuntimeError("No solution found")
