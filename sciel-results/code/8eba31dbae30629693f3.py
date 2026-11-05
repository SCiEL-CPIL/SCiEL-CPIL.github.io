"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Find 5 distinct integer denominators >= 5 summing to 1
    target = 1
    
    def dfs(remaining, min_den, count):
        if count == 5:
            if abs(remaining) < 1e-9:
                return list(used)
            return None
        
        # Pruning:
        # 1. 1/d <= remaining => d >= 1/remaining
        # 2. remaining - 1/d <= (5 - count - 1) / d  (max sum of rest)
        #    remaining <= (5 - count) / d
        #    d <= (5 - count) / remaining
        
        min_d = max(min_den, math.ceil(1 / remaining) if remaining > 0 else min_den)
        max_d = int((5 - count) / remaining) if remaining > 0 else min_d
        
        if min_d > max_d:
            return None

        for d in range(min_d, max_d + 1):
            if d in used:
                continue
            
            new_remaining = remaining - 1 / d
            if new_remaining < -1e-9:
                break
            
            used.add(d)
            res = dfs(new_remaining, d + 1, count + 1)
            if res:
                return res
            used.remove(d)
        
        return None

    used = set()
    # To speed up, we can fix the first term to 5
    used.add(5)
    res = dfs(1 - 1/5, 6, 1)
    if res:
        return res
    
    # Fallback if 5 is not in solution (unlikely for minimal count)
    used.remove(5)
    res = dfs(1, 5, 0)
    return res
# EVOLVE-BLOCK-END
