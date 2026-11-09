import time
from math import factorial
from bisect import bisect_right

def construct():
    start_time = time.time()
    target = factorial(12)
    
    # Precompute divisors of target greater than 12
    divisors = []
    limit = int(target**0.5)
    for i in range(13, limit + 1):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i:
                divisors.append(other)
    # Add target itself
    divisors.append(target)
    divisors = sorted(set(divisors))
    
    # Filter to only those > 12 (already done by range start, but target is added)
    # Ensure all are > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # We want to minimize the largest factor.
    # Iterate M through valid_divisors.
    for M in valid_divisors:
        if time.time() - start_time > 50:
            return [target]
        
        # Find index of M in valid_divisors
        # Candidates are valid_divisors[0 : idx+1]
        idx_M = bisect_right(valid_divisors, M)
        candidates = valid_divisors[:idx_M]
        
        def dfs(rem, last_factor, current_set):
            if rem == 1:
                return current_set
            
            if len(current_set) >= 12:
                return None
            
            # Find next factor x
            # x must be in candidates
            # x > last_factor
            # x <= rem
            # rem % x == 0
            
            # Optimization: if rem is prime and rem > last_factor and rem in candidates, 
            # we can return [rem] if rem <= M.
            
            for x in candidates:
                if x <= last_factor:
                    continue
                if x > rem:
                    break
                if rem % x != 0:
                    continue
                
                # Pruning: 
                # If rem // x > 1, it must be factorable into numbers > x and <= M.
                # If rem // x is prime and rem // x <= x, fail?
                # If rem // x == 1, handled by base case.
                
                res = dfs(rem // x, x, current_set + [x])
                if res:
                    return res
            return None

        solution = dfs(target, 12, [])
        if solution:
            return solution
            
    return [target]
