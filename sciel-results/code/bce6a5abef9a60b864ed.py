import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50
    
    denoms = list(range(2, 49))
    # Sort by value descending to maximize sum quickly
    # 1/2, 1/3, ..., 1/48
    items = [(1/d, i) for i, d in enumerate(denoms)]
    items.sort(key=lambda x: -x[0])
    
    N = len(items)
    values = [x[0] for x in items]
    indices = [x[1] for x in items]
    
    # Precompute suffix sums
    suffix = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]
        
    best_sum = Fraction(0)
    best_idxs = []
    
    # DFS
    # idx: current item index
    # curr_sum: sum of selected items so far
    # curr_idxs: list of selected indices so far
    def dfs(idx, curr_sum, curr_idxs):
        nonlocal best_sum, best_idxs
        
        if time.time() > deadline:
            return
            
        # If current sum is already >= 1, invalid
        if curr_sum >= 1:
            return
            
        # Update best if valid and better
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_idxs = curr_idxs[:]
            
        if idx == N:
            return
            
        # Pruning:
        # Even if we take all remaining items, can we beat best_sum?
        # We need a sum strictly less than 1.
        # If curr_sum + suffix[idx] <= best_sum, we can't get a strictly larger sum.
        if curr_sum + suffix[idx] <= best_sum:
            return
            
        # Option 1: Skip current item
        dfs(idx + 1, curr_sum, curr_idxs)
        
        # Option 2: Take current item
        new_sum = curr_sum + values[idx]
        if new_sum < 1:
            curr_idxs.append(indices[idx])
            dfs(idx + 1, new_sum, curr_idxs)
            curr_idxs.pop()
            
    dfs(0, Fraction(0), [])
    
    if not best_idxs:
        return [0] # Fallback 1/2
        
    return best_idxs
