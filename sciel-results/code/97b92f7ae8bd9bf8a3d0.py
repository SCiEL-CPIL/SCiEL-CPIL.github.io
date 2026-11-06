import math
import time

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = math.lcm(*denominators)
    V = [L // d for d in denominators]
    
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + V[i]
        
    best_deficit = L
    best_indices = []
    
    # Greedy init
    current_sum = 0
    temp_indices = []
    for i in range(n):
        if current_sum + V[i] < L:
            current_sum += V[i]
            temp_indices.append(i)
    if temp_indices:
        best_deficit = L - current_sum
        best_indices = temp_indices[:]
        
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Backtracking variables
    current_indices = [0] * n
    current_depth = 0
    timed_out = False
    
    def dfs(idx, current_sum_val):
        nonlocal best_deficit, best_indices, current_depth, timed_out
        
        if timed_out:
            return
        
        if time.time() > deadline:
            timed_out = True
            return
            
        # Pruning
        if current_sum_val + suffix[idx] <= L - best_deficit:
            return
        
        if idx == n:
            if current_sum_val < L:
                deficit = L - current_sum_val
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = current_indices[:current_depth]
            return
            
        # Option 1: Include idx
        if current_sum_val + V[idx] < L:
            current_indices[current_depth] = idx
            current_depth += 1
            dfs(idx + 1, current_sum_val + V[idx])
            current_depth -= 1
            
            if timed_out:
                return
                
        # Option 2: Exclude idx
        dfs(idx + 1, current_sum_val)
        
    dfs(0, 0)
    
    return best_indices
