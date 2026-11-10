import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # LCM of 1..24
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 25))
    
    # V[i] corresponds to element (i+1)
    # V[i] = L / (i+1)
    V = [L // (i + 1) for i in range(24)]
    
    best_set = ()
    
    def dfs(idx, current_set, current_sums):
        nonlocal best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning
        if len(current_set) + (24 - idx) <= len(best_set):
            return
            
        if idx == 24:
            if len(current_set) > len(best_set):
                best_set = current_set
            return
        
        # Option 1: Include idx
        v = V[idx]
        valid = True
        
        # Check if v is in current_sums (violation for new element)
        if v in current_sums:
            valid = False
        else:
            # Check if any existing element k is formed by v + subset of current_set
            for k in current_set:
                target_diff = V[k] - v
                if target_diff < 0:
                    continue
                if target_diff in current_sums:
                    valid = False
                    break
        
        if valid:
            # Generate new sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
            
            dfs(idx + 1, current_set + (idx,), new_sums)
        
        # Option 2: Exclude idx
        dfs(idx + 1, current_set, current_sums)

    # Start search
    # To improve pruning, we can try to order elements.
    # However, the index in V corresponds to the element value.
    # We can permute the order of processing.
    # Let's process elements in descending order of value (largest first)?
    # Or ascending?
    # Large values (small reciprocals) are less likely to be sums of others?
    # Actually, 1/1 is large. 1/24 is small.
    # Sums of small reciprocals are small.
    # It's hard to say which order is better for pruning.
    # Let's stick to natural order 1..24 (idx 0..23).
    
    dfs(0, (), {0})
    
    # Convert best_set (indices) to values
    if not best_set:
        return [1]
        
    return [idx + 1 for idx in best_set]
