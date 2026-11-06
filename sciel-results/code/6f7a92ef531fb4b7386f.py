import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Values as integers
    values = [L // d for d in denominators]
    
    # Suffix sums
    suf = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + values[i]
        
    # Greedy initial solution
    best_sum = 0
    best_indices = []
    curr_sum = 0
    curr_indices = []
    for i in range(n):
        if curr_sum + values[i] < L:
            curr_sum += values[i]
            curr_indices.append(i)
    best_sum = curr_sum
    best_indices = curr_indices
    
    current_indices = []
    
    def dfs(idx, current_sum):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
            
        if current_sum >= L:
            return
            
        if current_sum + suf[idx] <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
        
        # Include
        new_sum = current_sum + values[idx]
        if new_sum < L:
            current_indices.append(idx)
            dfs(idx + 1, new_sum)
            current_indices.pop()
            
        # Exclude
        dfs(idx + 1, current_sum)
        
    dfs(0, 0)
    
    return best_indices
