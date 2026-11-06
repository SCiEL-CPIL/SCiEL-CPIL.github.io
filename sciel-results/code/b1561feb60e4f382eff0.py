import time
from fractions import Fraction
from math import gcd
from functools import lru_cache

def construct():
    deadline = time.time() + 55  # safety margin for 90s wall-clock limit
    
    target = Fraction(5, 31)
    target_num, target_den = 5, 31
    
    # We know [8, 31, 248] works: 1/8 + 1/31 + 1/248 = 5/31
    best = [8, 31, 248]
    best_max = 248
    
    def verify(dens):
        s = Fraction(0, 1)
        for d in dens:
            s += Fraction(1, d)
        return s == target
    
    # Strategy: For a given max denominator C, check if there exists a valid
    # Egyptian fraction expansion of 5/31 using distinct denominators in [1, C].
    # We iterate C from the theoretical lower bound upward.
    
    # Lower bound: 1/d1 + 1/d2 + ... = 5/31. The smallest possible d1 is 7 (since 1/6 > 5/31).
    # So C >= 7. But we need at least one term, so C >= 7.
    # Actually, we need to check: can we do it with C < 248?
    
    # Key insight: For a fixed C, we can use DFS to find if any expansion exists
    # with all denominators <= C. We want the smallest such C.
    
    # Fast 2-term solver: given n/d, find distinct a, b <= C such that 1/a + 1/b = n/d
    # (n*a - d)*(n*b - d) = d^2
    def two_term(n, d, min_val, max_val):
        """Find distinct a, b with min_val <= a < b <= max_val, 1/a + 1/b = n/d."""
        # (n*a - d)*(n*b - d) = d^2
        # Let x = n*a - d, y = n*b - d, then x*y = d^2
        # a = (x + d)/n, b = (y + d)/n
        # Need x, y positive divisors of d^2, x <= y (so a <= b)
        # a >= min_val, b <= max_val, a != b
        # x = n*a - d >= n*min_val - d
        # y = n*b - d <= n*max_val - d
        # Also x*y = d^2, so x >= d^2/(n*max_val - d) if n*max_val - d > 0
        
        results = []
        d2 = d * d
        # x must satisfy: n*min_val - d <= x <= d (since x <= y and x*y = d^2 means x <= d)
        # Also x must divide d^2
        # And (x + d) % n == 0
        # And a = (x+d)//n >= min_val
        # And b = (y+d)//n where y = d^2//x, b <= max_val, b > a
        
        x_min = n * min_val - d
        if x_min < 1:
            x_min = 1
        x_max = d  # since x <= y and x*y = d^2
        
        # Also need y = d^2/x <= n*max_val - d, so x >= d^2/(n*max_val - d)
        if n * max_val - d > 0:
            x_min2 = (d2 + (n * max_val - d) - 1) // (n * max_val - d)  # ceil
            if x_min2 > x_min:
                x_min = x_min2
        else:
            return None  # can't fit b
        
        # Enumerate divisors x of d^2 in [x_min, x_max]
        # To be efficient, iterate x from x_min to x_max and check divisibility
        # But d^2 can be large. Better: enumerate divisors of d^2.
        
        # For small d (which is typical in our search), this is fine.
        # Let's just iterate.
        for x in range(x_min, x_max + 1):
            if d2 % x != 0:
                continue
            y = d2 // x
            if (x + d) % n != 0:
                continue
            a = (x + d) // n
            if (y + d) % n != 0:
                continue
            b = (y + d) // n
            if a < min_val or b > max_val:
                continue
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if a < min_val or b > max_val:
                continue
            results.append((a, b))
        
        return results if results else None
    
    # DFS for k-term expansion with all denominators in [min_d, C]
    def dfs(rem, min_d, C, depth, used):
        """
        rem: Fraction to represent
        min_d: minimum denominator allowed
        C: maximum denominator allowed
        depth: number of terms remaining
        used: set of denominators already used
        Returns a list of denominators if found, else None.
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            return [] if rem == 0 else None
        
        if depth == 1:
            if rem.numerator == 1:
                d = rem.denominator
                if min_d <= d <= C and d not in used:
                    return [d]
            return None
        
        if depth == 2:
            pairs = two_term(rem.numerator, rem.denominator, min_d, C)
            if pairs:
                for a, b in pairs:
                    if a not in used and b not in used:
                        return [a, b]
            return None
        
        # depth >= 3
        # 1/d < rem => d > 1/rem = rem.denominator / rem.numerator
        # d >= min_d
        # Also, the remaining (depth-1) terms each >= 1/C, so rem - 1/d >= (depth-1)/C
        # => 1/d <= rem - (depth-1)/C
        # => d >= 1 / (rem - (depth-1)/C)
        
        rem_num = rem.numerator
        rem_den = rem.denominator
        
        # d must satisfy: 1/d < rem, so d > rem_den / rem_num
        d_lower = max(min_d, rem_den // rem_num + 1)
        if rem_den % rem_num == 0:
            d_lower = max(min_d, rem_den // rem_num + 1)
        
        # Upper bound on d from the remaining sum constraint:
        # rem - 1/d >= (depth-1) * 1/C
        # 1/d <= rem - (depth-1)/C
        # If rem - (depth-1)/C <= 0, no solution
        if rem <= (depth - 1) / C:
            return None
        d_upper = C
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            if d in used:
                continue
            
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
            
            used.add(d)
            result = dfs(new_rem, d + 1, C, depth - 1, used)
            used.discard(d)
            
            if result is not None:
                return [d] + result
        
        return None
    
    # Search for the minimum C
    # Start from C = 7 (theoretical minimum) up to best_max - 1
    # For each C, try depths 1 through min(C, 256)
    
    found_better = False
    for C in range(7, best_max):
        if time.time() > deadline - 5:
            break
        
        # Try increasing number of terms
        # Minimum terms: at least ceil(1/(C * target)) but that's not quite right
        # Actually, the max sum with k terms all <= C is k/C (if all were 1/C, but they must be distinct)
        # The max sum with k distinct denominators <= C is 1/1 + 1/2 + ... but we need them >= 1
        # Actually we need sum = 5/31 ≈ 0.161. With denominators >= 7, max sum is 1/7 + 1/8 + ... 
        # Let's just try depths from 1 to 10 (or more if time permits)
        
        for depth in range(1, 12):
            if time.time() > deadline - 5:
                break
            
            # Quick check: can depth terms with denominators in [7, C] sum to 5/31?
            # Max possible sum: 1/7 + 1/8 + ... (largest distinct denominators starting from 7)
            # Actually the largest sum is using the smallest denominators: 1/7 + 1/8 + ... 
            # Wait, we want to achieve exactly 5/31. The max sum with depth terms from [7,C] 
            # is 1/7 + 1/8 + ... + 1/(7+depth-1) if depth <= C-7, else some other combination.
            # The min sum is 1/C + 1/(C-1) + ... + 1/(C-depth+1)
            
            # For efficiency, just try the DFS
            result = dfs(target, 7, C, depth, set())
            if result is not None:
                if verify(result):
                    best = result
                    best_max = C
                    found_better = True
                    break
        
        if found_better:
            # We found a better C, continue searching for even smaller? No, we're going up.
            # Actually we want the MINIMUM C, so once we find one at C, that's our answer
            # since we're iterating from low to high.
            break
    
    # If we didn't find anything better, return the known solution
    if not found_better:
        # Double-check our known solution
        if verify(best):
            return best
        # Fallback: try a few more depths with the original best_max
        for depth in range(4, 8):
            if time.time() > deadline - 2:
                break
            result = dfs(target, 7, best_max, depth, set())
            if result is not None and verify(result):
                return result
    
    return best
