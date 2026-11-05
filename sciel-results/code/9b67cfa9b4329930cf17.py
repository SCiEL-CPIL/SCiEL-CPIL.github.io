import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # LCM of 1 to 24
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # V[i] = L / i
    V = [0] * 25
    for i in range(1, 25):
        V[i] = L // i
        
    best_set = []
    
    def dfs(S, sums_S, candidates):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(S) > len(best_set):
            best_set = list(S)
            
        # Pruning: If len(S) + len(candidates) <= len(best_set), we can't improve.
        if len(S) + len(candidates) <= len(best_set):
            return
            
        for i, x in enumerate(candidates):
            vx = V[x]
            
            # Check 1: vx not in sums_S
            if vx in sums_S:
                continue
                
            # Check 2: For all y in S, V[y] - vx not in sums_S (if > 0)
            valid = True
            for y in S:
                target = V[y] - vx
                if target > 0 and target in sums_S:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # If valid, add x
            new_S = S + [x]
            # Update sums_S
            new_sums = set(sums_S)
            for s in sums_S:
                new_sums.add(s + vx)
                
            # Recurse with remaining candidates
            remaining_candidates = candidates[:i] + candidates[i+1:]
            
            dfs(new_S, new_sums, remaining_candidates)
            
    # Initial candidates: 1 to 24
    # Ascending order (1 to 24) tends to find large sets quickly because small integers are compatible.
    initial_candidates = list(range(1, 25))
    
    dfs([], {0}, initial_candidates)
    
    if not best_set:
        return [1]
        
    return best_set
