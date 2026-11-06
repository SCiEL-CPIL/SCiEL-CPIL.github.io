import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Suffix sums for pruning
    suffix = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    # Precompute conflict pairs for each x
    # conflict_pairs[x] = list of (a, b) such that a < b < x and a + b = 2x
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(3, N + 1):
        # a + b = 2x, a < b < x
        # b = 2x - a
        # a < 2x - a => 2a < 2x => a < x
        # 2x - a < x => x < a
        # So a must be such that x < a < 2x is false?
        # Wait. a < b < x.
        # b = 2x - a.
        # a < 2x - a => a < x.
        # 2x - a < x => x < a.
        # Contradiction?
        # Let's re-evaluate.
        # AP: a, b, x.
        # a + x = 2b => b = (a+x)/2.
        # We need a < b < x.
        # a < (a+x)/2 => 2a < a+x => a < x. (True)
        # (a+x)/2 < x => a+x < 2x => a < x. (True)
        # So any a < x such that (a+x) is even forms a valid b.
        # And b = (a+x)/2 is automatically < x and > a.
        # So we just iterate a from 1 to x-1.
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if a < b: # b > a is guaranteed if a < x
                    conflict_pairs[x].append((a, b))

    # Greedy initialization: "Remove if conflict with smaller elements"
    # Start with all elements, remove x if it forms AP with a,b < x already kept.
    # This is equivalent to:
    # S = []
    # for x in 1..N:
    #    if no a,b in S such that a,b,x form AP:
    #       S.append(x)
    # This is the standard forward greedy.
    
    # Let's try the "Backward Greedy" or "Conflict Removal" from full set.
    # Start with mask having all bits set.
    # Iterate x from N down to 1?
    # If we iterate x from N down to 1, and x is in S, check if it forms AP with a,b in S.
    # Since we are going down, a,b > x are already decided.
    # If x forms AP with a,b > x, we might remove x.
    # This preserves small numbers?
    # If we remove x, we keep a,b.
    # If a,b are large, we keep large numbers. Bad for weight.
    # So we should iterate x from 1 to N, and if x conflicts with a,b < x, we remove x.
    # This is the same as forward greedy: keep x only if it doesn't conflict with previous.
    
    # Let's stick to forward greedy for initial bound, it's fast.
    mask_init = 0
    for x in range(1, N + 1):
        valid = True
        for a, b in conflict_pairs[x]:
            if (mask_init & (1 << a)) and (mask_init & (1 << b)):
                valid = False
                break
        if valid:
            mask_init |= (1 << x)
            
    # Calculate score for initial mask
    best_score = 0.0
    for i in range(1, N + 1):
        if mask_init & (1 << i):
            best_score += weights[i]
            
    best_mask = mask_init

    # DFS
    # To speed up, we can use a global best_score and update it.
    # We also need to ensure we don't exceed time.
    
    def dfs(idx, current_sum, mask):
        nonlocal best_score, best_mask
        
        if time.time() > deadline:
            return
            
        if current_sum + suffix[idx] <= best_score:
            return
            
        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_mask = mask
            return
            
        x = idx
        
        # Option 1: Include x
        # Check validity
        valid = True
        for a, b in conflict_pairs[x]:
            if (mask & (1 << a)) and (mask & (1 << b)):
                valid = False
                break
                
        if valid:
            new_mask = mask | (1 << x)
            dfs(idx + 1, current_sum + weights[x], new_mask)
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum, mask)

    dfs(1, 0.0, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result
