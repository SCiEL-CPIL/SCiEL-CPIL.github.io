import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 60.0

def _dfs(remaining, min_d, used, current_max, best_max_holder):
    """
    DFS to find a solution with max denominator < best_max_holder['val'].
    """
    if remaining == 0:
        if current_max < best_max_holder['val']:
            best_max_holder['val'] = current_max
            best_max_holder['sol'] = list(used)
        return True
    
    if current_max >= best_max_holder['val']:
        return False
        
    # Pruning: If remaining is too small to be formed by any number of terms >= min_d
    # Max possible sum with infinite terms starting from min_d is infinite? No, sum 1/d diverges.
    # But we have a cap on max denominator? No, the cap is best_max_holder.
    # So all d must be <= best_max_holder['val'].
    # Max sum with terms in [min_d, best_max] is sum_{d=min_d}^{best_max} 1/d.
    # If remaining > that sum, prune.
    # This calculation is expensive.
    
    # Better pruning:
    # If remaining < 1 / best_max_holder['val']:
    #   We need a term 1/d such that 1/d <= remaining? No.
    #   If remaining is very small, we need large d.
    #   If 1/best_max < remaining, we MUST use a denominator d < best_max?
    #   No, if remaining < 1/best_max, we can use d > best_max? No, we are capped at best_max.
    #   If we are capped at best_max, the smallest term we can add is 1/best_max.
    #   If remaining < 1/best_max, we cannot form it with any single term.
    #   But we can use multiple terms?
    #   Sum of multiple terms 1/d_i.
    #   If remaining < 1/best_max, and all d_i <= best_max, then 1/d_i >= 1/best_max > remaining.
    #   So sum of any positive number of such terms > remaining.
    #   So if remaining < 1/best_max, it is IMPOSSIBLE to form remaining with denominators <= best_max.
    #   Wait, this logic is flawed.
    #   Example: remaining = 0.1. best_max = 10. 1/10 = 0.1. OK.
    #   remaining = 0.05. best_max = 10. 1/10 = 0.1 > 0.05.
    #   Can we form 0.05 with d <= 10?
    #   1/20 = 0.05, but 20 > 10.
    #   1/11 + ...? No, d must be <= 10.
    #   Smallest sum with one term is 1/10.
    #   So if remaining < 1/best_max, we CANNOT form it.
    #   So prune if remaining < 1 / best_max_holder['val'].
    
    if remaining < Fraction(1, best_max_holder['val']):
        return False

    # Pruning: If remaining is too large
    # Max sum we can add with remaining terms?
    # We don't know how many terms we have left.
    # But we know we can add at most MAX_TERMS - len(used) terms.
    # The largest sum we can add is if we use the smallest available denominators.
    # Min available is min_d.
    # Max sum = sum_{i=0}^{k-1} 1 / (min_d + i) where k = MAX_TERMS - len(used).
    # If remaining > max_sum, prune.
    
    k = MAX_TERMS - len(used)
    if k <= 0:
        return False
        
    max_sum = Fraction(0, 1)
    for i in range(k):
        max_sum += Fraction(1, min_d + i)
        
    if remaining > max_sum:
        return False

    # Iterate next denominator
    # d must be >= min_d
    # d must be <= best_max_holder['val']
    # Also, to avoid duplicates and ensure order, d > last used?
    # We maintain `used` as a set, and pass `min_d` as the lower bound.
    # To ensure distinctness and increasing order (for pruning efficiency), we assume d > previous d.
    
    for d in range(min_d, best_max_holder['val'] + 1):
        if d in used:
            continue
            
        term = Fraction(1, d)
        if term > remaining:
            # Since d is increasing, term is decreasing.
            # If term > remaining, we need a larger d.
            continue
            
        new_rem = remaining - term
        if new_rem < 0:
            continue
            
        if d > current_max:
            new_max = d
        else:
            new_max = current_max
            
        # Recurse
        used.add(d)
        if _dfs(new_rem, d + 1, used, new_max, best_max_holder):
            used.remove(d)
            return True
        used.remove(d)
        
    return False

def construct():
    start_time = time.time()
    
    # Known solution
    best_sol = [8, 31, 248]
    best_max = 248
    
    # Try to find better solution
    # We search for max denom < 248
    # We can try limits 247, 246, ... down to some bound
    # But DFS with limit 247 might be slow if no solution exists.
    # Let's try limit 247 first.
    
    # Optimization: 
    # If we find a solution with max 200, we can try 199, etc.
    # But 247 is the immediate target.
    
    # To speed up, we can use the fact that 5/31 is fixed.
    # Let's run DFS with cap = 247.
    
    # If time runs out, return best_sol.
    
    # We need to be careful with recursion depth. MAX_TERMS 256 is fine.
    
    # One issue: The DFS above iterates d from min_d to cap.
    # If min_d is small (e.g. 7), and cap is 247, that's 240 iterations.
    # Next level: 239 iterations.
    # This can explode.
    # We need stronger pruning.
    
    # Pruning: 
    # If remaining > 0, we MUST pick a d such that 1/d <= remaining.
    # So d >= ceil(1/remaining).
    # Let min_d_effective = max(min_d, ceil(1/remaining)).
    
    # Also, if we pick d, the new remaining is R' = R - 1/d.
    # We need to form R' with denominators > d.
    # The max sum we can form with denominators > d and cap C is sum_{i=d+1}^C 1/i.
    # If R' > that sum, prune.
    
    # Let's implement this improved DFS.
    
    def dfs_optimized(remaining, min_d, used, current_max, cap):
        if remaining == 0:
            return list(used)
            
        if current_max >= cap:
            return None
            
        if remaining < Fraction(1, cap):
            return None
            
        # Lower bound for d
        # 1/d <= remaining => d >= 1/remaining
        min_d_eff = max(min_d, (1 / remaining).numerator // (1 / remaining).denominator + 1 if (1 / remaining).numerator % (1 / remaining).denominator != 0 else (1 / remaining).numerator // (1 / remaining).denominator)
        # Simpler: ceil(1/remaining)
        # 1/remaining is Fraction.
        inv_rem = 1 / remaining
        min_d_eff = max(min_d, (inv_rem.numerator + inv_rem.denominator - 1) // inv_rem.denominator)
        
        # Upper bound for d?
        # We need to leave enough sum for the rest.
        # Max sum with k terms starting from d+1 is bounded.
        # But we don't know k.
        
        for d in range(min_d_eff, cap + 1):
            if d in used:
                continue
                
            term = Fraction(1, d)
            if term > remaining:
                continue
                
            new_rem = remaining - term
            if new_rem < 0:
                continue
                
            # Pruning: Can we form new_rem with denominators in [d+1, cap]?
            # Max possible sum with infinite terms is infinite.
            # But we have a cap on terms? No, MAX_TERMS.
            # If new_rem > 0, we need at least 1 more term.
            # Smallest term we can add is 1/cap.
            # If new_rem < 1/cap, we can't form it?
            # No, we can form it with multiple terms?
            # If new_rem < 1/cap, and all future d' > d, then 1/d' < 1/d < 1/cap? No, d' >= d+1 > d.
            # So 1/d' < 1/d.
            # If 1/d < new_rem, we are fine.
            # If new_rem is very small, we might need a very large d'.
            # But d' <= cap.
            # So if new_rem < 1/cap, it is IMPOSSIBLE to form new_rem with any number of terms <= cap?
            # Wait. If new_rem = 0.001 and cap = 1000. 1/1000 = 0.001.
            # We can use d'=1000.
            # If new_rem = 0.0009. 1/1000 = 0.001 > 0.0009.
            # We can't use 1/1000.
            # We can use 1/1100? No, cap is 1000.
            # So if new_rem < 1/cap, we CANNOT form it.
            if new_rem > 0 and new_rem < Fraction(1, cap):
                continue
                
            # Check if new_rem can be formed by sum of 1/d' where d' in [d+1, cap].
            # Max sum with MAX_TERMS-len(used)-1 terms.
            # This is complex. Let's rely on the recursion.
            
            used.add(d)
            if time.time() - start_time > DEADLINE - 2:
                used.remove(d)
                return None
                
            res = dfs_optimized(new_rem, d + 1, used, max(current_max, d), cap)
            used.remove(d)
            
            if res:
                return res
                
        return None

    # Try to find solution with cap 247
    # Start with empty used, min_d = 1?
    # But d must be >= 7 effectively.
    # We can start min_d = 7.
    
    sol = dfs_optimized(TARGET, 7, set(), 0, 247)
    
    if sol:
        return sol
    else:
        return best_sol
