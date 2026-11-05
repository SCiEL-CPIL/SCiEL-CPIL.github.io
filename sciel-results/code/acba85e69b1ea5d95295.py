from fractions import Fraction
from functools import lru_cache

def construct():
    # We know min terms is 8.
    # Search for k=8, 9, 10...
    
    def search(rem, min_d, depth):
        # rem: Fraction
        # min_d: int, minimum denominator for next term
        # depth: int, number of terms left to choose
        
        if depth == 1:
            # rem must be 1/d
            if rem.denominator == 1:
                d = rem.numerator
                if d >= min_d:
                    return [d]
            else:
                # Check if rem is unit fraction
                # rem = n/d. If n==1, d is the denom.
                if rem.numerator == 1:
                    d = rem.denominator
                    if d >= min_d:
                        return [d]
            return None

        # Bounds for d
        # 1/d <= rem  => d >= 1/rem
        d_min = int(1 / rem) + 1
        if d_min < min_d:
            d_min = min_d
            
        # Upper bound logic:
        # rem - 1/d <= sum_{j=1}^{depth-1} 1/(d+j)
        # For large d, 1/d is small, rem-1/d approx rem.
        # Max sum of depth-1 terms starting at d+1 is approx (depth-1)/d.
        # So rem <= 1/d + (depth-1)/d = depth/d
        # d <= depth / rem
        # This is a loose upper bound.
        d_max = int(depth / rem) + 10 # Add buffer
        
        # Refine d_max:
        # We need rem - 1/d <= sum_{j=1}^{depth-1} 1/(d+j)
        # Let's just iterate d from d_min to d_max and check feasibility quickly.
        
        for d in range(d_min, d_max + 1):
            # Check if 1/d <= rem
            if Fraction(1, d) > rem:
                break # d too small? No, 1/d > rem means d < 1/rem. 
                # Wait, if d increases, 1/d decreases.
                # We started at d_min where 1/d_min <= rem.
                # So for d > d_min, 1/d < rem. This is always valid for the "overshoot" check.
                
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
                
            # Check if new_rem can be formed by depth-1 terms > d
            # Max possible sum for depth-1 terms starting from d+1:
            # S_max = sum(1/(d+j) for j in 1..depth-1)
            # If new_rem > S_max, then d is too large (we didn't take enough value).
            # Actually, if d is large, 1/d is small, new_rem is large.
            # The subsequent terms are even smaller. So we can't reach new_rem.
            # So if new_rem > S_max, we should increase d?
            # No, if we increase d, 1/d decreases, new_rem increases. S_max decreases.
            # So if new_rem > S_max at some d, it will be worse for larger d.
            # So we should break?
            # Let's check:
            # d small -> 1/d large -> new_rem small. S_max large. Likely feasible.
            # d large -> 1/d small -> new_rem large. S_max small. Likely infeasible.
            # So if we find a d where new_rem > S_max, all larger d will also fail.
            # So we can break.
            
            # Calculate S_max efficiently or just use a bound
            # S_max < (depth-1) / (d+1)
            if new_rem > Fraction(depth - 1, d + 1):
                break
                
            # Recursive call
            res = search(new_rem, d + 1, depth - 1)
            if res:
                return [d] + res
                
        return None

    for k in range(8, 20):
        res = search(Fraction(1, 1), 5, k)
        if res:
            return res
            
    raise RuntimeError("No solution found")
