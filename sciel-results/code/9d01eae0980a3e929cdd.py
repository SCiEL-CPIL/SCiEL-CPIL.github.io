import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline for safety (90s limit)
    
    target = factorial(12)
    
    # Generate all divisors of target
    divisors = set()
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
        i += 1
        
    # Filter divisors > 12
    candidates = [d for d in divisors if d > 12]
    candidates.sort()
    
    best_max = target
    best_set = [target]
    
    def dfs(idx, curr_prod, curr_max, chosen):
        nonlocal best_max, best_set
        
        if curr_prod == target:
            if curr_max < best_max:
                best_max = curr_max
                best_set = list(chosen)
            return
        
        if len(chosen) == 12:
            return
            
        if curr_max >= best_max:
            return
            
        # Pruning: check if remaining product can be achieved
        remaining = target // curr_prod
        
        # If remaining is 1, we need curr_prod == target which is handled above
        if remaining <= 1:
            return
            
        # If remaining itself is a candidate and not already chosen, we could take it
        # But we need to check if it's distinct and > 12
        
        # Stronger pruning: the smallest possible product of k more distinct factors > curr_max
        # is (curr_max+1)*(curr_max+2)*...*(curr_max+k)
        # If even that exceeds remaining, we can't complete
        
        # For now, let's just iterate but with better ordering
        
        for i in range(idx, len(candidates)):
            d = candidates[i]
            
            if curr_prod * d > target:
                break
                
            if target % (curr_prod * d) != 0:
                continue
                
            # If d is already in chosen, skip (shouldn't happen due to idx, but safety)
            if d in chosen:
                continue
                
            chosen.append(d)
            dfs(i + 1, curr_prod * d, max(curr_max, d), chosen)
            chosen.pop()
            
            if time.time() > deadline:
                return

    chosen = []
    dfs(0, 1, 0, chosen)
    
    return best_set
