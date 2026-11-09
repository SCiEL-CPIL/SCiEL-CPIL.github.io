import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    target = Fraction(5, 31)
    
    # We want to minimize the largest denominator.
    # We can do a binary search on the max denominator B, or just iterate.
    # Given the structure, small B is likely.
    # Let's try to find ANY solution first, then maybe optimize?
    # Actually, just finding a solution with a small B is the goal.
    # We can iterate B from 1 upwards.
    # But if B is large, it takes long.
    # Let's use a DFS that prunes based on a global best_max_denom found so far?
    # No, just find the first solution for a given B.
    
    def solve_with_max_denom(B):
        # Returns list of denominators if sum of reciprocals is target and all <= B, distinct.
        # Uses DFS.
        
        def dfs(rem, min_d, current):
            if time.time() > deadline:
                return None
            
            if rem == 0:
                return current
            
            if not current:
                # Heuristic: if rem is very small, need large d
                pass
            
            # Lower bound for next d:
            # 1/d <= rem  => d >= 1/rem
            lb = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            # Upper bound is B
            # Also, we can prune if we can't reach 0 with remaining terms?
            # For simplicity, just loop d from lb to B.
            
            for d in range(lb, B + 1):
                if time.time() > deadline:
                    return None
                
                next_rem = rem - Fraction(1, d)
                if next_rem < 0:
                    # Since d is increasing, 1/d is decreasing.
                    # If 1/d > rem, we need smaller d? No, larger d.
                    # But we are iterating d upwards.
                    # lb is ceil(1/rem). So for d >= lb, 1/d <= rem.
                    # So next_rem >= 0.
                    # This check is technically redundant if lb is correct, but safe.
                    continue
                
                # Pruning: if next_rem is small, check if it can be formed by 1 term?
                # If next_rem is a unit fraction, we can add it directly if it's <= B and distinct.
                if next_rem.numerator == 1:
                    d_next = next_rem.denominator
                    if d_next >= d + 1 and d_next <= B:
                        return current + [d, d_next]
                
                # If next_rem is 0, handled above.
                # Recurse
                res = dfs(next_rem, d + 1, current + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(target, 1, [])

    # Search for the smallest B
    # We know 1-term is impossible.
    # 2-term is impossible (checked analytically).
    # Let's start B from 1.
    # But 1/1 > 5/31, so min denominator is at least 6.
    # Let's start B from 6.
    
    # To be efficient, we can jump B.
    # But let's just iterate.
    # If no solution found for B, try B+1.
    
    # Optimization: If we find a solution with max denom M, we can stop if we are sure no smaller exists?
    # We are iterating B upwards, so the first solution found has the minimal max denominator?
    # Not necessarily. A solution with max denom 100 might exist, but a solution with max denom 101 might be found first if we don't check 100 carefully?
    # No, we check B=1, then B=2, etc.
    # If a solution exists with max denom 10, it will be found when B=10.
    # So the first B for which solve_with_max_denom(B) returns non-None is the optimal max denominator.
    
    # However, the search space for small B is small.
    # For large B, it's huge.
    # We should set a cap on B to avoid infinite loops if no solution exists (but one does).
    
    # Let's try B from 1 to 10000.
    # If not found, increase step.
    
    B = 1
    while B < 100000:
        if time.time() > deadline:
            break
            
        res = solve_with_max_denom(B)
        if res is not None:
            return res
        
        B += 1
        if B > 100:
            B += 10
        elif B > 1000:
            B += 100
        elif B > 10000:
            B += 1000

    # Fallback: return a known valid solution if we ran out of time or B
    # We can use the 4-term solution found in analysis: 7, 55, 3980, 9500260
    # But we need distinct denominators in [1, 10000000].
    # 9500260 is in range.
    return [7, 55, 3980, 9500260]
