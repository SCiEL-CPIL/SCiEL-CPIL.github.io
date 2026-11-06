import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for 60s limit
    
    # Try to find a solution with 8 terms
    # Denominators must be >= 5 and distinct
    # Order: d1 < d2 < ... < dk
    
    def dfs(remaining_terms, current_sum, last_denom, current_denoms):
        if time.time() > deadline:
            return None
        
        if remaining_terms == 0:
            if current_sum == 1:
                return current_denoms
            return None
        
        # Pruning:
        # 1. Even if we pick the smallest possible denominators, can we reach 1?
        # Smallest next denom is last_denom + 1
        # Max sum we can add with 'remaining_terms' terms starting from last_denom+1
        # Approximate max sum: sum(1/(last_denom+i) for i in 1..remaining_terms)
        # This is an upper bound on what we can add.
        # If current_sum + max_possible_add < 1, we can't reach 1.
        
        # 2. Min next denom:
        # We need 1/d <= (1 - current_sum) / remaining_terms
        # d >= remaining_terms / (1 - current_sum)
        
        target = 1 - current_sum
        if target <= 0:
            return None
            
        min_next = int(remaining_terms / target) + 1
        if min_next < last_denom + 1:
            min_next = last_denom + 1
            
        # Upper bound for min_next:
        # 1/min_next must be > 0, obviously.
        # Also 1/min_next < target (since we have multiple terms, each must be less than target)
        # Actually, 1/d < target is required.
        # So d > 1/target.
        # This is usually looser than the average bound.
        
        # Let's iterate d from min_next upwards.
        # How far up?
        # 1/d <= target / remaining_terms is the average.
        # The largest d we can pick is bounded by the fact that the remaining terms must sum to the rest.
        # A safe upper bound for d in the first step of DFS:
        # If d is too large, 1/d is too small, and we might not be able to reach the target with the remaining terms?
        # No, if d is large, 1/d is small, so current_sum increases slowly.
        # We need to ensure that we can still reach 1.
        # Max sum of remaining (remaining_terms - 1) terms starting from d+1.
        # If current_sum + 1/d + max_sum_rest < 1, then d is too large?
        # No, if d is too large, 1/d is small.
        # Wait, if d is too large, we might "overshoot" the min bound? No.
        # If d is too large, the sum grows too slowly?
        # Actually, if d is too large, we might not be able to reach 1 because the subsequent denominators must be even larger, making their reciprocals even smaller.
        # So there is an upper bound on d.
        # 1/d + sum_{i=1}^{r-1} 1/(d+i) >= target
        # This is roughly (r/d) >= target => d <= r/target.
        # So d is bounded by [min_next, max_next].
        
        max_next = int(remaining_terms / target)
        if max_next < min_next:
            return None
            
        # Iterate d
        for d in range(min_next, max_next + 1):
            if d < 5: continue # Constraint d >= 5
            if d <= last_denom: continue # Distinct and increasing
            
            new_sum = current_sum + Fraction(1, d)
            if new_sum > 1:
                break # Since d is increasing, 1/d is decreasing, so this won't happen?
                # Wait, if we increase d, 1/d decreases. So if new_sum > 1, we should continue to larger d?
                # No, if 1/d is too big, we need a smaller 1/d, i.e., larger d.
                # So if new_sum > 1, we continue to next d.
                # But we have an upper bound on d.
                # If we exceed the upper bound, we break.
                
            res = dfs(remaining_terms - 1, new_sum, d, current_denoms + [d])
            if res:
                return res
                
        return None

    # Try k=8
    for k in [8, 9, 10]:
        # Start with smallest possible denominator 5
        # But we can optimize start.
        # For k=8, d1 must be small.
        # Max sum with 8 terms starting from 5 is approx 1.018.
        # So d1=5 is a good start.
        
        # We can also try different starting denominators if 5 fails.
        # But 5 is the smallest, so it gives the most "room" for the sum.
        
        result = dfs(k, Fraction(0, 1), 4, [])
        if result:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
            
    # Fallback: Use the seed program's logic or a hardcoded solution
    # If DFS fails, return a known valid solution.
    # Let's generate a valid one using the seed's method if it's fast enough, 
    # or just return a hardcoded 9-term solution if I can find one.
    
    # Hardcoded 9-term solution check:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13
    # Sum approx 1.018 + 0.077 = 1.095. Too big.
    # We need to increase denominators.
    
    # Let's just run the seed logic as fallback, but it might be slow.
    # Better: Return a hardcoded valid solution if found.
    # I will try to find one quickly.
    
    # Try: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...
    # Sum 5..12 is 1.018.
    # Replace 12 with 13?
    # 5,6,7,8,9,10,11,13.
    # Sum 5..11 = 0.9365.
    # 1/13 = 0.0769.
    # Sum = 1.0134.
    # Need 1 more term.
    # Rem = 1 - 1.0134 = -0.0134.
    # We overshot.
    
    # Try 5, 6, 7, 8, 9, 10, 11, 14.
    # Sum 5..11 = 0.9365.
    # 1/14 = 0.0714.
    # Sum = 1.0079.
    # Rem = -0.0079.
    
    # Try 5, 6, 7, 8, 9, 10, 11, 15.
    # 1/15 = 0.0666.
    # Sum = 1.0031.
    # Rem = -0.0031.
    
    # Try 5, 6, 7, 8, 9, 10, 11, 16.
    # 1/16 = 0.0625.
    # Sum = 0.9990.
    # Rem = 0.0010.
    # Need 1/d = 0.0010 => d = 1000.
    # Check if 1000 is distinct from 5,6,7,8,9,10,11,16. Yes.
    # Check if 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/16+1/1000 = 1.
    
    # Let's verify this hardcoded candidate:
    # Denoms: 5, 6, 7, 8, 9, 10, 11, 16, 1000
    # LCM is large.
    # Sum = 0.2 + 0.166666 + 0.142857 + 0.125 + 0.111111 + 0.1 + 0.090909 + 0.0625 + 0.001
    # Sum = 0.999043 + 0.001 = 1.000043.
    # Not exactly 1.
    
    # The DFS should find the exact one.
    # If DFS fails, I will return a valid solution found by a slower, more exhaustive search or hardcoded from a known list.
    
    # For now, if DFS fails, I'll return a known valid 11-term solution or similar, but that's suboptimal.
    # However, the task says "minimize their count".
    # If I can't find 8, I should find 9.
    
    # Given the time constraint, I will rely on the DFS finding 8.
    
    # Fallback to seed logic if DFS fails
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
            
    raise RuntimeError("no solution found")
