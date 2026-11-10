"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Target: sum = 1, min denom = 5
    # Search for min k
    
    def dfs(remaining, min_denom, terms_left, current_path):
        if time.time() > deadline:
            return None
            
        if terms_left == 0:
            if remaining == 0:
                return list(current_path)
            else:
                return None
        
        # Pruning: Max possible sum with 'terms_left' terms starting from min_denom
        # Sum_{i=1}^{terms_left} 1/(min_denom + i)
        max_sum = Fraction(0, 1)
        for i in range(1, terms_left + 1):
            max_sum += Fraction(1, min_denom + i)
        if max_sum < remaining:
            return None
        
        # Lower bound for next denominator d
        # 1/d < remaining  => d > 1/remaining
        # Also d >= min_denom
        # d must be integer
        if remaining == 0:
            return None # Should not happen if terms_left > 0 and we check exact match
            
        # d > 1/remaining
        # If remaining > 1, impossible, but remaining <= 1 always.
        # d >= floor(1/remaining) + 1
        # Careful with Fraction to int conversion
        d_min = min_denom
        if remaining > 0:
            # 1/remaining is a Fraction
            # We want d such that 1/d < remaining
            # d > 1/remaining
            # Let x = 1/remaining. d > x.
            # d_min_candidate = int(x) + 1
            inv_rem = Fraction(1, 1) / remaining
            d_min_candidate = int(inv_rem) + 1
            if d_min_candidate > d_min:
                d_min = d_min_candidate
        
        # Upper bound for d
        # We need to pick 'terms_left' denominators d, d2, ...
        # The smallest possible sum for the remaining 'terms_left' terms (including current d)
        # is achieved by large denominators, but we are bounded by the *max* sum of the *subsequent* terms.
        # After picking d, remaining sum is rem' = remaining - 1/d.
        # We have terms_left - 1 terms left.
        # Max sum of those is Sum_{i=1}^{terms_left-1} 1/(d+i).
        # We need Sum_{i=1}^{terms_left-1} 1/(d+i) >= remaining - 1/d.
        # Let's iterate d from d_min upwards until this condition fails.
        
        d = d_min
        while True:
            # Check if we can finish with this d
            rem_after = remaining - Fraction(1, d)
            if rem_after < 0:
                break # 1/d > remaining, so 1/(d+1) will be even smaller? No, 1/d decreases.
                # Wait. If 1/d > remaining, then rem_after < 0.
                # We need rem_after >= 0.
                # If 1/d > remaining, then d < 1/remaining.
                # But we set d_min > 1/remaining, so 1/d < remaining.
                # So rem_after > 0.
                
            if rem_after == 0:
                if terms_left == 1:
                    current_path.append(d)
                    return list(current_path)
                else:
                    # We have 0 remaining sum but terms left. Impossible.
                    pass
            
            # Check max sum of remaining terms
            if terms_left > 1:
                max_sub_sum = Fraction(0, 1)
                for i in range(1, terms_left):
                    max_sub_sum += Fraction(1, d + i)
                if max_sub_sum < rem_after:
                    # Even with the smallest possible next denominators (d+1, d+2...), 
                    # we can't reach the remaining sum.
                    # Since increasing d increases the denominators d+i, the max sum decreases.
                    # So for larger d, it will be even harder.
                    break
            
            # Recurse
            current_path.append(d)
            res = dfs(rem_after, d + 1, terms_left - 1, current_path)
            current_path.pop()
            
            if res:
                return res
            
            # Increment d
            d += 1
            # Optimization: if d becomes very large, we might timeout.
            # But the break condition above should catch it.
            
        return None

    # Try k from 5 to 15
    for k in range(5, 16):
        # Initial call
        # remaining = 1
        # min_denom = 5
        # terms_left = k
        res = dfs(Fraction(1, 1), 5, k, [])
        if res:
            return res
            
    # Fallback: return a known valid solution if search fails (should not happen)
    # Example: 5, 6, 7, 42, 140, ... (Need to verify)
    # Let's just return a large set from the original logic if this fails, 
    # but the search should find k=5 or 6.
    raise RuntimeError("No solution found within time limit")
# EVOLVE-BLOCK-END
