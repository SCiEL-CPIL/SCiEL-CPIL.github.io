import math
import time

def construct():
    """
    Returns a list of 1..12 distinct integers > 12 with product 12!,
    minimizing the largest factor.
    """
    target = math.factorial(12)
    
    # Generate divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    candidates = [d for d in divisors if d > 12]
    candidates.sort()
    
    best_max = float('inf')
    best_solution = []
    
    # To speed up, we can precompute which candidates divide the remaining product?
    # No, simple DFS is likely fast enough given the constraints.
    
    def dfs(start_idx, current_prod, current_max, path):
        nonlocal best_max, best_solution
        
        # Pruning 1: If current product exceeds target
        if current_prod > target:
            return
            
        # Pruning 2: If current max is already >= best found, no point continuing
        if current_max >= best_max:
            return
            
        # Check if we reached the target
        if current_prod == target:
            if len(path) > 0: # At least one factor
                if current_max < best_max:
                    best_max = current_max
                    best_solution = path[:]
            return

        # We need to pick at least one more factor if current_prod != target
        # Iterate through candidates starting from start_idx
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Pruning 3: If x itself is >= best_max, we can't improve the max.
            # Since candidates are sorted, all subsequent x will be larger.
            if x >= best_max:
                break
                
            # Pruning 4: If current_prod * x > target, we can't use x or any larger x
            if current_prod * x > target:
                break
                
            # Pruning 5: Divisibility check
            if target % (current_prod * x) != 0:
                continue
                
            # Recursive step
            # The new max is max(current_max, x)
            dfs(i + 1, current_prod * x, max(current_max, x), path + [x])
            
    # Initial call
    # We can start with empty product 1, max 0 (or 12, since factors > 12)
    # path is empty
    dfs(0, 1, 0, [])
    
    if best_solution:
        return best_solution
    else:
        # Fallback: Return the single factor 12! if no decomposition found (should not happen)
        # But 12! > 12, so [12!] is a valid solution with max 12!
        return [target]
