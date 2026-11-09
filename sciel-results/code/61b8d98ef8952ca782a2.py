import time
from math import factorial

def construct():
    start_time = time.time()
    target = factorial(12)
    
    # Precompute divisors of target greater than 12
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    # Add target itself if it's > 12 (it is)
    if target not in divisors:
        divisors.append(target)
    divisors = sorted(set(divisors))
    
    # We want to minimize the largest factor.
    # Iterate M from 13 upwards.
    # For each M, check if a valid set exists with max factor <= M.
    
    # Upper bound for M: 
    # If we use 1 factor, M = target.
    # We can just loop M from 13 to target, but break early.
    
    # To speed up, we can iterate through possible max factors M 
    # which must be a divisor of target? 
    # Not necessarily. The max factor in the set MUST be a divisor of the target 
    # because it is one of the factors.
    # So we only need to test M values that are divisors of target.
    
    # Let's sort divisors and iterate through them as potential M.
    # But wait, M is the max factor. The set must be composed of divisors <= M.
    # So iterating M through divisors is correct.
    
    for M in divisors:
        if time.time() - start_time > 50:
            # Fallback: return a known valid solution or the single factor
            # The single factor [target] is always valid.
            return [target]
        
        # DFS to find a set of distinct divisors from divisors <= M 
        # that multiply to target.
        # We restrict candidates to divisors <= M.
        
        def dfs(rem, last_factor, current_set):
            if rem == 1:
                return current_set
            
            # Find next factor x
            # x must be in divisors
            # x > last_factor
            # x <= M
            # x <= rem
            # rem % x == 0
            
            # Optimization: 
            # If rem > 1, we need at least one more factor.
            # The smallest possible next factor is last_factor + 1 (but must be in divisors).
            # If the product of the smallest possible remaining factors exceeds rem, fail?
            # This is complex. Simple search is likely fast enough.
            
            for x in divisors:
                if x <= last_factor:
                    continue
                if x > M:
                    break # divisors are sorted
                if x > rem:
                    break
                if rem % x != 0:
                    continue
                
                # Additional prune:
                # If rem // x > 1, it must be factorable into numbers > x and <= M.
                # If rem // x is prime and rem // x <= x, it's impossible.
                # If rem // x == 1, we are done (handled by base case in next call).
                
                res = dfs(rem // x, x, current_set + [x])
                if res:
                    return res
            return None

        solution = dfs(target, 12, [])
        if solution:
            return solution
            
    return [target]
