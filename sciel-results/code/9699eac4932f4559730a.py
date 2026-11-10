import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Target: find 8 distinct denominators >= 5 summing to 1
    # We know k=8 is the minimum possible.
    
    def dfs(remaining_num, remaining_den, min_denom, current, max_depth):
        """
        remaining: remaining sum as fraction num/den
        min_denom: minimum allowed denominator
        current: list of chosen denominators
        max_depth: max allowed depth
        """
        if time.time() > deadline:
            return None
        
        if remaining_num == 0:
            if len(current) == max_depth:
                return current
            return None
        
        if len(current) >= max_depth:
            return None
        
        # If remaining is a unit fraction, we can add it directly
        if remaining_den == 1:
            d = remaining_num
            if d >= min_denom and d >= 5 and d not in current:
                if len(current) + 1 <= max_depth:
                    # Check if we can fill the rest? No, if remaining is unit fraction,
                    # we just add it. But we need exactly max_depth terms?
                    # The problem says "minimize their count".
                    # So if we find a solution with k terms, we return it.
                    # The DFS is called with max_depth=k.
                    # If we find a solution with fewer terms, we should have found it in a previous iteration.
                    # So if we are in max_depth=8, we want exactly 8 terms.
                    # But if remaining_num/remaining_den is a unit fraction 1/d,
                    # we can add d and stop. This gives len(current)+1 terms.
                    # If len(current)+1 < max_depth, we have a solution with fewer terms.
                    # But we iterate max_depth from 2 upwards.
                    # So if we are at max_depth=8, we are looking for 8 terms.
                    # If we find a solution with 7 terms, we would have returned it in max_depth=7 loop.
                    # So here, if remaining is unit fraction, and len(current)+1 == max_depth, return.
                    # If len(current)+1 < max_depth, it means we found a shorter solution,
                    # which should have been caught earlier. So we can return None or the solution.
                    # To be safe, return the solution.
                    return current + [d]
        
        # Try adding the next denominator
        # 1/d <= remaining, so d >= 1/remaining = den/num
        min_d = max(min_denom, 5, remaining_den // remaining_num + 1)
        
        # Upper bound: 1/d >= remaining/(max_depth - len(current))
        # So d <= (max_depth - len(current)) / remaining
        max_terms_left = max_depth - len(current)
        max_d = (max_terms_left * remaining_den) // remaining_num
        
        for d in range(min_d, min(max_d + 1, 10000001)):
            if d in current:
                continue
            # new_remaining = remaining - 1/d
            # = (num*d - den) / (den*d)
            new_num = remaining_num * d - remaining_den
            if new_num < 0:
                break
            new_den = remaining_den * d
            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            result = dfs(new_num, new_den, d + 1, current + [d], max_depth)
            if result is not None:
                return result
        
        return None

    # Try increasing number of terms
    # We know k=8 is minimum, so start from 8?
    # But to be safe, start from 2.
    for max_terms in range(2, 10):
        result = dfs(1, 1, 5, [], max_terms)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(result) == len(set(result)) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: return a known valid solution if found
    # Or use the DP fallback from the seed
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
