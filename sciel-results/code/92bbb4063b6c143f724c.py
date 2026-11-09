import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit (or 90s if that's the limit, but 50 is safe)
    
    # 1. Calculate 12!
    target = math.factorial(12)  # 479001600
    
    # 2. Generate all divisors of target efficiently using prime factorization
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):      # 2^a
        for b in range(6):   # 3^b
            for c in range(3): # 5^c
                for d in range(2): # 7^d
                    for e in range(2): # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Initial best solution: single factor [target]
    best_max = target
    best_factors = [target]
    
    # DFS to find a decomposition with max factor < current best_max
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = current_factors[:]
            return
        
        # We can have at most 12 factors
        if len(current_factors) >= 12:
            return
        
        # Pruning: if current_max is already >= best_max, no point continuing
        # Since we add factors in increasing order (due to sorted valid_divs and start_idx),
        # current_max is non-decreasing along the path.
        if current_max >= best_max:
            return
        
        # Another pruning: if remaining > 1 and we can't add any more factors that divide it
        # and are <= remaining, we fail. But the loop below handles this.
        
        # Also, if the product of the smallest possible remaining factors can't reach 'remaining', 
        # it's hard to prune simply. But we can check if 'remaining' is not divisible by any 
        # valid divisor >= current last factor and <= remaining.
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            
            # Since valid_divs is sorted, if d > remaining, no further d can divide remaining
            # (except if remaining is 1, which is handled above).
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            
            # If new_max >= best_max, we can prune.
            # Since d is increasing, new_max will only stay same or increase.
            # So if new_max >= best_max, all subsequent d will also lead to new_max >= best_max.
            if new_max >= best_max:
                break
            
            current_factors.append(d)
            dfs(remaining // d, i + 1, current_factors, new_max)
            current_factors.pop()
            
            # If we found a better solution, best_max might have decreased.
            # The loop condition `if new_max >= best_max: break` will handle pruning for next iterations.
            
    dfs(target, 0, [], 0)
    
    return best_factors
