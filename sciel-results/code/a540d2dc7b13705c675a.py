import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Reciprocals as Fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Suffix sums for branch and bound: suffix[i] = sum of recips[i:]
    # This allows us to compute an upper bound on the sum we can achieve
    # starting from index i, assuming we pick all remaining items.
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[i]
    
    best_deficit = Fraction(1, 1)
    best_indices = []
    best_sum = Fraction(0)
    
    def update_best(indices, s):
        nonlocal best_deficit, best_indices, best_sum
        if s < 1:
            d = 1 - s
            if d < best_deficit:
                best_deficit = d
                best_indices = list(indices)
                best_sum = s
    
    # Initialize with greedy solution
    cur_sum = Fraction(0)
    cur_indices = []
    for i in range(n):
        if cur_sum + recips[i] < 1:
            cur_sum += recips[i]
            cur_indices.append(i)
    update_best(cur_indices, cur_sum)
    
    # Branch and Bound DFS
    # We visit indices in order 0..n-1.
    # State: current index i, current sum s, current list of indices.
    # Pruning:
    # 1. If s >= 1, prune (invalid).
    # 2. If s + suffix[i] <= best_sum, prune (cannot beat best_sum, since we want to maximize sum < 1).
    #    Note: We are looking for the largest sum strictly less than 1.
    #    Deficit = 1 - sum. Minimizing deficit is equivalent to maximizing sum subject to sum < 1.
    
    def dfs(i, s, indices):
        nonlocal best_deficit, best_indices, best_sum, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Pruning 1: If current sum is already >= 1, invalid branch.
        if s >= 1:
            return
        
        # Pruning 2: Upper bound check.
        # The maximum possible sum from this state is s + suffix[i].
        # If this upper bound is <= best_sum, we cannot find a better solution (larger sum < 1).
        # Wait, we need sum < 1. If s + suffix[i] >= 1, it's possible we can get closer to 1.
        # If s + suffix[i] < 1, then even picking all remaining items won't reach 1.
        # In that case, the best we can do in this subtree is s + suffix[i].
        # If s + suffix[i] <= best_sum, we prune.
        # If s + suffix[i] > best_sum, we might find a better sum.
        
        # Actually, a tighter bound:
        # We want to maximize s' such that s' < 1 and s' is reachable from s by adding subset of recips[i:].
        # Upper bound on achievable sum is s + suffix[i].
        # If s + suffix[i] <= best_sum, prune.
        
        if s + suffix[i] <= best_sum:
            return
        
        # If we've considered all items
        if i == n:
            update_best(indices, s)
            return
        
        # Option 1: Skip item i
        dfs(i + 1, s, indices)
        
        # Option 2: Take item i
        # Check if taking item i keeps sum < 1
        if s + recips[i] < 1:
            indices.append(i)
            dfs(i + 1, s + recips[i], indices)
            indices.pop()
    
    # Start DFS
    dfs(0, Fraction(0), [])
    
    return best_indices
