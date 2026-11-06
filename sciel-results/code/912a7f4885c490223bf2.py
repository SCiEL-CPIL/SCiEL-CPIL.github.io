import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Helper: max possible sum of 'terms' distinct unit fractions with denominators >= min_d
    # Each term <= 1/(min_d + i) for i=0..terms-1, but that's an overestimate.
    # Tighter: the sum is at most 1/min_d + 1/(min_d+1) + ... + 1/(min_d+terms-1)
    def max_sum(terms, min_d):
        if terms <= 0:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for i in range(terms):
            s += Fraction(1, min_d + i)
        return s

    # Helper: min possible sum of 'terms' distinct unit fractions with denominators >= min_d
    # Each term >= 1/(10000000), but that's too weak. Better: the sum is at least terms/10000000
    # Actually for pruning we mainly need max_sum.

    # Try to find a solution with exactly k terms using DFS
    def find_solution(k):
        path = []
        
        def dfs(rem, min_d, k_left):
            if time.time() > deadline:
                return False
            
            if k_left == 0:
                return rem == Fraction(0, 1)
            
            # Pruning 1: max possible sum of k_left terms with denominators >= min_d
            if rem > max_sum(k_left, min_d):
                return False
            
            # Pruning 2: min possible sum - if rem is too small, we can't make it
            # The minimum sum of k_left distinct unit fractions with denominators >= min_d
            # is at least k_left / 10000000 (very weak)
            # Better: if rem < k_left * Fraction(1, 10000000), return False
            # But this is almost never triggered. Skip.
            
            # Find valid range for d
            # 1/d <= rem => d >= 1/rem
            if rem <= 0:
                return False
            
            d_min = max(min_d, math.ceil(1 / float(rem)))
            if d_min < 5:
                d_min = 5
            
            # Upper bound: we need rem - 1/d <= max_sum(k_left - 1, d + 1)
            # i.e., 1/d >= rem - max_sum(k_left - 1, d + 1)
            # For large d, 1/d is small, rem - max_sum(...) is positive and ~rem
            # So d can't be too large.
            # Approximate: rem - (k_left-1)/(d+1) <= 1/d
            # rem <= 1/d + (k_left-1)/(d+1) ~ k_left/d for large d
            # So d ~ k_left / rem
            d_max_est = int(k_left / float(rem)) + 100
            d_max = min(d_max_est, 10000000)
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return False
                
                # 1/d must be <= rem
                if Fraction(1, d) > rem:
                    continue
                
                rem_new = rem - Fraction(1, d)
                
                if rem_new == 0:
                    if k_left == 1:
                        path.append(d)
                        return True
                    else:
                        continue
                
                if rem_new < 0:
                    break  # d too small? No, we have 1/d <= rem, so rem_new >= 0
                
                path.append(d)
                if dfs(rem_new, d + 1, k_left - 1):
                    return True
                path.pop()
            
            return False
        
        if dfs(Fraction(1, 1), 5, k):
            return path
        return None

    # Try k from 8 upward
    for k in range(8, 257):
        if time.time() > deadline:
            break
        result = find_solution(k)
        if result is not None:
            return result

    # Fallback: use a known valid solution
    # Let's construct one: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/1260 + 1/2520
    # This doesn't work. Let me find a valid one.
    # 1/5 + 1/6 = 11/30. Remainder: 19/30.
    # 19/30 = 1/3 + 4/30 = 1/3 + 2/15. But 3 < 5.
    # 19/30 = 1/4 + 7/30. 4 < 5.
    # 19/30 = 1/5 + 14/30 = 1/5 + 7/15. 7/15 = 1/3 + 4/15. 3 < 5.
    # 7/15 = 1/4 + 13/60. 4 < 5.
    # 7/15 = 1/5 + 2/3. 2/3 = 1/3 + 1/3. Not distinct.
    
    # Let me use a different approach. 
    # 1/5 + 1/6 + 1/7 + 1/42 = (42+35+30+5)/210 = 112/210 = 8/15
    # Remainder: 7/15
    # 7/15 = 1/5 + 2/3. 2/3 = 1/4 + 1/12. 4 < 5.
    # 2/3 = 1/5 + 7/15. Circular.
    # 7/15 = 1/6 + 23/30. 23/30 = 1/3 + 1/30. 3 < 5.
    # 23/30 = 1/4 + 17/60. 4 < 5.
    # 23/30 = 1/5 + 14/30 = 1/5 + 7/15. Same as before.
    
    # Let me try a larger known decomposition.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 - excess
    # Sum = 28271/27720. Excess = 551/27720.
    # This is messy.
    
    # Instead, let me just return a solution found by a simple greedy that works:
    # Use the greedy algorithm with constraint d >= 5
    path = []
    rem = Fraction(1, 1)
    min_d = 5
    while rem > 0 and len(path) < 256:
        if time.time() > deadline:
            break
        # Find smallest d >= min_d such that 1/d <= rem and rem - 1/d can be formed
        # Simple greedy: pick d = ceil(1/rem)
        d = max(min_d, math.ceil(1 / float(rem)))
        if d > 10000000:
            break
        if Fraction(1, d) > rem:
            d += 1
        path.append(d)
        rem -= Fraction(1, d)
        min_d = d + 1
    
    if rem == 0 and len(path) >= 1 and len(path) <= 256:
        return path
    
    # Last resort: raise error
    raise RuntimeError("No solution found")
