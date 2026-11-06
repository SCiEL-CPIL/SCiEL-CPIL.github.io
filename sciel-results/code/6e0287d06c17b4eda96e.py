import time
from fractions import Fraction

def construct():
    """
    Find 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1.
    Minimize their count.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def dfs(rem_num, rem_den, last_d, terms, depth, max_depth):
        """
        rem: rem_num / rem_den
        last_d: previous denominator chosen
        terms: list of denominators chosen so far
        """
        if time.time() > deadline:
            return None
        
        if rem_num == 0:
            return terms if len(terms) > 0 else None
        
        if depth == max_depth:
            return None
        
        # Lower bound for d:
        # 1/d <= rem => d >= 1/rem = rem_den / rem_num
        # d > last_d
        # d >= 5
        min_d = max(last_d + 1, 5)
        # Also d must be such that 1/d <= rem
        # d >= rem_den / rem_num
        import math
        lb_from_val = math.ceil(rem_den / rem_num) if rem_num > 0 else min_d
        min_d = max(min_d, lb_from_val)
        
        # Upper bound for d:
        # We have (max_depth - depth) slots left, including current.
        # Let k = max_depth - depth - 1 be the number of slots after this one.
        # The remaining value after picking d is rem - 1/d.
        # This remaining value must be <= the sum of the largest possible k unit fractions
        # with denominators > d. The largest such fractions are 1/(d+1), 1/(d+2), ..., 1/(d+k).
        # So: rem - 1/d <= sum_{i=1}^{k} 1/(d+i)
        # rem <= 1/d + sum_{i=1}^{k} 1/(d+i)
        # rem <= sum_{i=0}^{k} 1/(d+i)
        
        k = max_depth - depth - 1
        # If k < 0, we are at the last term, handled by depth check? No, depth < max_depth.
        # If depth == max_depth - 1, k=0.
        # Then rem <= 1/d => d <= 1/rem.
        
        # We need to find max d such that sum_{i=0}^{k} 1/(d+i) >= rem
        # Since sum decreases as d increases, we can binary search or just iterate down?
        # Actually, as d increases, 1/d decreases, so the sum decreases.
        # We want the LARGEST d such that the condition holds?
        # No, we iterate d from min_d upwards.
        # The condition `sum_{i=0}^k 1/(d+i) >= rem` will hold for small d and fail for large d.
        # So there is a MAX d.
        
        # Let's estimate max_d.
        # Approximation: (k+1)/d >= rem => d <= (k+1)/rem = (k+1)*rem_den/rem_num
        max_d_approx = ((max_depth - depth) * rem_den) // rem_num
        
        # Refine max_d by checking the exact sum bound.
        # We can start from max_d_approx and go down until the bound holds?
        # Or just use the approximation and let the recursion fail if it's too loose?
        # The approximation is usually a safe upper bound (it overestimates the sum capacity for a given d? 
        # No, sum 1/(d+i) < (k+1)/d. So if (k+1)/d < rem, then sum < rem.
        # So d must be <= (k+1)/rem.
        # So max_d_approx is a valid upper bound.
        
        max_d = min(max_d_approx, 10000000)
        
        if max_d < min_d:
            return None

        for d in range(min_d, max_d + 1):
            # Check if 1/d <= rem
            # rem_num/rem_den >= 1/d
            # d * rem_num >= rem_den
            if d * rem_num < rem_den:
                continue
            
            # Subtract 1/d
            # new_rem = rem_num/rem_den - 1/d
            # = (d*rem_num - rem_den) / (d*rem_den)
            
            new_num = d * rem_num - rem_den
            new_den = d * rem_den
            
            if new_num == 0:
                return terms + [d]
            
            if new_num < 0:
                break # Should not happen due to check above, but safe
            
            # Recurse
            result = dfs(new_num, new_den, d, terms + [d], depth + 1, max_depth)
            if result is not None:
                return result
        
        return None

    # Try to find solutions with increasing number of terms
    # Theoretical minimum is likely around 11-15 for this constraint.
    # Start from a reasonable number. 10 might be too tight/impossible.
    for k in range(10, 20):
        if time.time() > deadline:
            break
        # Start with remaining = 1/1
        result = dfs(1, 1, 4, [], 0, k)
        if result is not None:
            return result

    # Fallback: If DFS fails, try a greedy approach that doesn't strictly minimize count
    # but ensures validity.
    # Greedy: pick smallest d > last such that 1/d <= rem.
    # This might not terminate quickly or might use too many terms, but we have a deadline.
    # However, the DFS above should find a solution for k=11 or 12 quickly.
    
    # If we reach here, we failed to find a minimal solution within time.
    # Return a known valid solution if we have one hardcoded?
    # Let's try to construct one quickly using a simple identity.
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/147 + 1/350 + 1/392 + 1/399 + 1/7140 ...
    # This is risky.
    
    # Let's just raise an error if we can't find it, but the DFS should work.
    raise RuntimeError("no valid representation found")
