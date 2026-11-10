import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Target sum is 1
    target = Fraction(1, 1)
    min_denom = 5
    
    # Try increasing number of terms
    # Theoretical min is 8 based on sum of first 8 reciprocals starting from 5
    # Sum(1/5 to 1/12) > 1, so 8 is possible in terms of magnitude.
    # We need to check if an exact solution exists.
    
    def dfs(target, min_d, remaining, current_list):
        # Check time
        if time.time() > deadline:
            return None
            
        if remaining == 0:
            if target == 0:
                return current_list
            return None
            
        if target <= 0:
            return None
            
        # Calculate max possible sum with 'remaining' terms starting from min_d
        # This is sum(1/(min_d + i) for i in range(remaining))
        # If target > max_sum, impossible
        max_sum = Fraction(0, 1)
        for i in range(remaining):
            max_sum += Fraction(1, min_d + i)
            if max_sum >= target:
                break
        if max_sum < target:
            return None
            
        # Determine range for next denominator d
        # 1. d >= min_d
        # 2. 1/d <= target  => d >= 1/target
        # 3. We must be able to reach (target - 1/d) with (remaining - 1) terms >= d+1
        #    Max sum of remainder: sum(1/(d + j) for j in 1..remaining-1)
        #    So target - 1/d <= max_rem_sum
        
        # Lower bound for d
        lb = max(min_d, (1 + target.numerator - 1) // target.denominator) # ceil(1/target)
        
        # Upper bound for d
        # We iterate d upwards.
        # Stop when it becomes impossible to reach target with remaining terms.
        # Condition: target - 1/d > max_sum_of(remaining-1, d+1)
        
        # Optimization: We can estimate an upper bound for d.
        # If d is very large, 1/d is small, target - 1/d is close to target.
        # The max sum of remaining terms decreases as d increases.
        # So there is a max d.
        
        # Let's just iterate d from lb upwards, with a cap.
        # A safe cap: if 1/d < target / remaining, then even if we pick the smallest possible remaining denominators, 
        # wait, no.
        
        # Let's iterate.
        # To avoid infinite loop or too large range, we can stop if:
        # target - 1/d > sum(1/(d+j) for j in 1..remaining-1)
        
        # Estimate max d.
        # For large d, sum(1/(d+j)) approx (remaining-1)/d.
        # target - 1/d approx target.
        # target > (remaining-1)/d  => d > (remaining-1)/target.
        # So d is roughly bounded by (remaining-1)/target + something.
        
        # Let's use a loop with a check.
        
        # Calculate max_rem_sum for a given d
        def max_rem_sum(d, rem):
            s = Fraction(0, 1)
            for j in range(1, rem + 1):
                s += Fraction(1, d + j)
            return s

        # Iterate d
        # We need to be careful with the upper limit of d to ensure performance.
        # If d is too large, 1/d is tiny.
        # The condition `target - 1/d > max_rem_sum(d, remaining-1)` will become true eventually.
        
        # Start d from lb
        # We can also impose a hard cap on d based on the deadline or a large number.
        # But the mathematical bound should prune it.
        
        d = lb
        # Safety: don't iterate too many if something goes wrong
        # But for k=8, lb is small.
        
        while True:
            if time.time() > deadline:
                return None
                
            # Check if we can reach target with this d
            # 1. 1/d must be <= target
            if Fraction(1, d) > target:
                return None # Should not happen if lb is correct, but good check
                
            rem_target = target - Fraction(1, d)
            
            if rem_target < 0:
                return None
                
            if rem_target == 0:
                if remaining == 1:
                    return current_list + [d]
                else:
                    return None # Can't have 0 sum with positive terms
            
            # Check if rem_target is achievable with remaining-1 terms starting from d+1
            # Max sum of remaining-1 terms starting from d+1
            max_rem = max_rem_sum(d, remaining - 1)
            
            if rem_target > max_rem:
                # If we can't reach the target even with the smallest denominators for the rest,
                # then increasing d will make 1/d smaller, rem_target larger, and max_rem smaller.
                # So it will never work for larger d.
                return None
                
            # Recursive step
            res = dfs(rem_target, d + 1, remaining - 1, current_list + [d])
            if res is not None:
                return res
                
            d += 1
            
            # Optional: break if d gets too large?
            # The condition `rem_target > max_rem` should trigger return None.
            # But what if rem_target <= max_rem for a long time?
            # For fixed remaining, as d increases, rem_target increases (towards target) and max_rem decreases.
            # So they will cross.
            
            # To prevent infinite loops in case of precision issues (though Fraction is exact),
            # we can add a counter or rely on the deadline.
            
            # Actually, if rem_target <= max_rem, it's a valid candidate to explore.
            # If it fails, we try next d.
            
            # Is there an upper bound on d?
            # If d > 1/rem_target, then 1/d < rem_target, which is fine.
            # But we need to check the max_rem condition.
            
            # Let's add a hard stop if d exceeds a reasonable limit, e.g., 10^7
            if d > 10000000:
                return None

    # Search for minimal k
    for k in range(8, 20): # 8 to 19
        # Clear any state if needed, but dfs is pure
        result = dfs(target, min_denom, k, [])
        if result is not None:
            return result
            
    # Fallback: return a known valid solution if search fails (should not happen)
    # The seed program finds one.
    # Let's just raise an error or return a large set if all else fails.
    # But 8 or 9 should be found.
    
    raise RuntimeError("No solution found within time limit")
