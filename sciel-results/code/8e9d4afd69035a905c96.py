"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    target = Fraction(5, 31)
    
    # We know 1,2,3,4,5 are too big individually.
    # So denominators must be >= 6.
    # We search for the minimum possible max denominator D.
    # We can iterate D from 6 upwards.
    
    # However, a more efficient search is to pick denominators one by one,
    # ensuring they are distinct and the sum matches.
    # To minimize the max, we should prefer smaller denominators?
    # No, smaller denominators contribute more to the sum.
    # We need to fill exactly 5/31.
    
    # Let's use a recursive search that tries to add the next denominator.
    # To keep the max small, we should bound the denominators.
    # We can try all combinations of size k for k=1 to 5, with max denom D.
    
    # Given the structure, let's try to find a solution with a small max denominator.
    # We'll iterate D from 6 to 100.
    # For each D, we check if any subset of {6..D} sums to 5/31.
    # This is a subset sum problem with rational numbers.
    
    # Optimization: 
    # If we fix D, we can use DP or DFS.
    # Since D is small, DFS is fine.
    
    def dfs(remaining, start_d, current_dens, max_d):
        if time.time() > deadline:
            return None
        
        if remaining == 0:
            return current_dens
        
        if remaining < 0:
            return None
        
        # Heuristic: The next denominator d must satisfy 1/d <= remaining
        # So d >= 1/remaining.
        min_d = max(start_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Also d cannot exceed max_d
        if min_d > max_d:
            return None
            
        for d in range(min_d, max_d + 1):
            next_rem = remaining - Fraction(1, d)
            if next_rem < 0:
                break
            # Pruning: if next_rem > 0, we need to fill it with denominators > d.
            # The max possible sum with denominators > d up to max_d is bounded.
            # But for small D, we just recurse.
            res = dfs(next_rem, d + 1, current_dens + [d], max_d)
            if res:
                return res
        return None

    # Try increasing max_d
    for max_d in range(6, 1000):
        # Start with the smallest possible denominator for the first term.
        # Since 1/6 > 5/31, we must start with d >= 7? 
        # No, 1/6 = 0.166 > 0.161. So 6 cannot be the first term.
        # But 6 could be a later term? No, terms are positive.
        # So all terms must be >= 7?
        # Wait, if we use 7, 1/7 = 0.142 < 0.161.
        # So we can start with 7.
        
        # Let's just start the DFS from the smallest valid d.
        # The smallest d such that 1/d <= target is d >= 31/5 = 6.2 -> 7.
        # So start_d = 7.
        
        res = dfs(target, 7, [], max_d)
        if res:
            return res
            
    # Fallback to seed-like behavior if not found
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
