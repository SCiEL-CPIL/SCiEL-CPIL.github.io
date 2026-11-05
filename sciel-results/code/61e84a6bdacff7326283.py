import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline for safety
    
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
    
    best_max = target # Initial best is the target itself (single factor)
    best_set = [target]
    
    # We can also add a heuristic: start with a good guess to prune early?
    # Maybe just run the DFS.
    
    # DFS
    # State: index in candidates, current_product, current_max
    # We want to find a set of distinct candidates from 'candidates'
    # such that product is target.
    
    def dfs(idx, curr_prod, curr_max):
        nonlocal best_max, best_set
        
        if curr_prod == target:
            if curr_max < best_max:
                best_max = curr_max
                best_set = list(current_choice) # Need to track choice
            return
        
        # If we have picked 12 factors, stop
        if len(current_choice) == 12:
            return
            
        # Pruning: if current max is already worse than best, stop
        if curr_max > best_max:
            return
            
        # Pruning: if curr_prod * smallest_possible_remaining > target? 
        # Actually, we just iterate.
        
        for i in range(idx, len(candidates)):
            d = candidates[i]
            
            # If adding d exceeds target, break
            if curr_prod * d > target:
                break
                
            # If d doesn't divide the remaining part, skip
            if target % (curr_prod * d) != 0:
                continue
                
            # Choose d
            current_choice.append(d)
            dfs(i + 1, curr_prod * d, max(curr_max, d))
            current_choice.pop()
            
            # Time check
            if time.time() > deadline:
                return

    current_choice = []
    dfs(0, 1, 0)
    
    return best_set
