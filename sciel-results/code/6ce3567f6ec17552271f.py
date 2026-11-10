from fractions import Fraction
from itertools import combinations

def construct():
    # Search for minimum number of distinct denominators >= 5 summing to 1
    # We know we need at least 6 terms (5 terms max sum < 1)
    # Let's do a BFS/DFS search
    
    # Try increasing number of terms
    for n in range(6, 20):
        # Use a search: pick denominators greedily or use backtracking
        # For small n, we can try combinations but that's expensive
        # Let's use a smarter approach: recursive search with pruning
        
        def search(remaining, min_denom, count, path):
            if count == 0:
                return path if remaining == 0 else None
            if count == 1:
                # Need 1/x = remaining, so x = 1/remaining
                if remaining > 0 and (1 / remaining) == int(1 / remaining):
                    x = int(1 / remaining)
                    if x >= max(min_denom, 5) and x >= 5:
                        return path + [x]
                return None
            
            # Upper bound: max possible sum with 'count' terms starting from min_denom
            # is sum of 1/min_denom, 1/(min_denom+1), ..., 1/(min_denom+count-1)
            max_possible = sum(Fraction(1, min_denom + i) for i in range(count))
            if max_possible < remaining:
                return None
            
            # Lower bound: min possible sum
            # We need remaining/count <= 1/min_denom for feasibility
            # Each term <= 1/min_denom, so count * 1/min_denom >= remaining
            
            # Try each possible next denominator
            # The next denominator d must satisfy: 1/d <= remaining (since we need positive sum)
            # and 1/d >= remaining / count (since remaining terms each at most 1/d... wait no)
            # Actually: 1/d must be at least remaining/count (if all terms equal)
            # and 1/d must be at most remaining/(1) if count=1, but for count>1:
            # 1/d <= remaining - sum of smallest (count-1) terms
            # Smallest (count-1) terms: 1/(max_denom), 1/(max_denom-1), ... but we don't have max
            # Let's just bound: 1/d <= remaining (trivial) and 1/d >= remaining/count
            
            # d >= max(min_denom, ceil(1/remaining) if count==1 else ceil(count/remaining))
            # Actually 1/d >= remaining/count means d <= count/remaining
            # And 1/d <= remaining means d >= 1/remaining
            
            lo = max(min_denom, 5)
            hi = min(10000000, int(count / remaining) if remaining > 0 else 10000000)
            
            if hi < lo:
                return None
            
            for d in range(lo, hi + 1):
                if remaining - Fraction(1, d) < 0:
                    continue
                # Check if remaining - 1/d can be represented with (count-1) terms
                # each >= d+1
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    continue
                # Upper bound check for remaining terms
                max_rest = sum(Fraction(1, d + 1 + i) for i in range(count - 1))
                if max_rest < new_remaining:
                    continue
                result = search(new_remaining, d + 1, count - 1, path + [d])
                if result is not None:
                    return result
            return None
        
        result = search(Fraction(1), 5, n, [])
        if result is not None:
            return result
    
    raise RuntimeError("no solution found")
