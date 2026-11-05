import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Nodes for small set
    smalls = list(range(2, 129))
    n = len(smalls)
    
    # Precompute conflicts in smalls
    # conflict[i][j] is True if lcm(smalls[i], smalls[j]) <= 256
    # We can use bitmasks for adjacency
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a = smalls[i]
            b = smalls[j]
            if math.lcm(a, b) <= 256:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                
    # Precompute for each small x, which large y it blocks
    # large_y is in [129, 256]
    # blocked_by[x] = list of y in [129, 256] such that x | y
    # Also sum of 1/y for those y
    blocked_sum = [0.0] * n
    blocked_mask_large = [0] * n # bitmask for large numbers? 128 numbers, fits in int
    # Let's index large numbers 0..127 corresponding to 129..256
    for i, x in enumerate(smalls):
        s = 0.0
        mask = 0
        for k, y in enumerate(range(129, 257)):
            if y % x == 0:
                s += 1.0 / y
                mask |= 1 << k
        blocked_sum[i] = s
        blocked_mask_large[i] = mask
        
    # We want to maximize:
    # Sum(1/x for x in S) + Sum(1/y for y in Large if not blocked by S)
    # Let Total_Large_Sum = sum(1/y for y in Large)
    # Score(S) = Total_Large_Sum + Sum(1/x for x in S) - Sum(1/y for y in Large blocked by S)
    # Note: if multiple x in S block same y, we should only subtract 1/y once.
    # So we need to track the union of blocked masks.
    
    total_large_sum = sum(1.0 / y for y in range(129, 257))
    
    best_score = 0.0
    best_set = []
    
    # Branch and Bound
    # State: 
    #   available_mask: bitmask of smalls still available
    #   current_mask: bitmask of smalls selected
    #   blocked_large_mask: bitmask of large numbers blocked by current selection
    #   current_small_sum: sum of 1/x for selected smalls
    
    # To prune, we can use an upper bound.
    # UB = current_small_sum + total_large_sum - (sum of 1/y for y in blocked_large_mask) 
    #      + (max possible additional small sum)
    # Max possible additional small sum is sum of 1/x for all available smalls.
    # This is a valid UB because adding smalls only increases small sum, and might increase blocked_large (decreasing score), so ignoring that penalty gives an upper bound.
    
    # Precompute sum of 1/x for subsets? No, too many.
    # Just iterate.
    
    def dfs(avail_mask, curr_small_sum, blocked_large_mask, curr_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        # Calculate current score
        # Sum of 1/y for large not blocked
        # We can maintain this incrementally or compute from mask.
        # Since n_large=128, mask is int.
        # Let's precompute sum for all masks? 2^128 impossible.
        # We can maintain a variable `current_large_sum`.
        # But we pass blocked_large_mask.
        # Let's just compute the score at the leaf or when we decide to stop.
        
        # Upper Bound
        # Max additional small sum
        # We can precompute for each mask? No.
        # Just sum 1/x for x in avail_mask.
        # This is expensive if done every step.
        # Optimization: precompute sum_1_over_x for all smalls.
        # And use a simple bound: curr_small_sum + sum(1/x for x in avail_mask) + total_large_sum
        # This is very loose.
        
        # Better:
        # The score is curr_small_sum + (total_large_sum - sum_blocked)
        # where sum_blocked is sum of 1/y for y in blocked_large_mask.
        # We can maintain sum_blocked incrementally.
        
        # Let's refactor to pass sum_blocked.
        
        # For now, let's just do a simple search with a decent bound.
        # Bound: curr_small_sum + sum_{x in avail} (1/x) + total_large_sum
        # This assumes we can add all available smalls and no large are blocked.
        # This is safe.
        
        # Calculate bound
        bound = curr_small_sum + total_large_sum
        for i in range(n):
            if avail_mask & (1 << i):
                bound += 1.0 / smalls[i]
                
        if bound <= best_score:
            return
            
        # If no more available
        if avail_mask == 0:
            score = curr_small_sum + (total_large_sum - sum_blocked)
            if score > best_score:
                best_score = score
                best_set = list(curr_set)
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick the one with highest 1/x or highest degree
        # Let's pick the first available for simplicity, or highest weight
        # To be efficient, pick the one that blocks the most large numbers?
        # Or just iterate.
        
        # To improve speed, we can use the property that if we don't pick i, we remove it from avail.
        # If we pick i, we remove it and its neighbors from avail, and update blocked mask.
        
        # Let's pick the available vertex with the largest 1/x (smallest x) to find good solutions early?
        # Or the one with highest "impact".
        # Let's just pick the lowest index available for determinism, or highest 1/x.
        
        # Find index of a vertex to branch on
        # Let's try the one with the highest 1/x (smallest value) among available
        # This often finds high weight solutions quickly.
        
        branch_idx = -1
        max_val = -1
        for i in range(n):
            if avail_mask & (1 << i):
                val = 1.0 / smalls[i]
                if val > max_val:
                    max_val = val
                    branch_idx = i
                    
        if branch_idx == -1:
            return
            
        x = smalls[branch_idx]
        
        # Option 1: Include x
        # Check if x conflicts with current set?
        # We maintain avail_mask such that all available are compatible with current set.
        # So x is compatible.
        
        new_avail = avail_mask & ~(adj[branch_idx] | (1 << branch_idx))
        new_curr_small_sum = curr_small_sum + 1.0 / x
        new_blocked_mask = blocked_large_mask | blocked_mask_large[branch_idx]
        # Update sum_blocked? 
        # We need to know how much sum_blocked increased.
        # delta = sum(1/y for y in (new_blocked_mask ^ blocked_large_mask))
        # We can precompute sum for masks? No.
        # We can just compute sum_blocked from mask at the end, or maintain it.
        # Maintaining it is better.
        # Let's pass sum_blocked as well.
        
        # Wait, I didn't include sum_blocked in the signature above.
        # Let's restart the dfs function definition with correct params.
        
        pass # Placeholder

    # Let's implement the DFS properly with sum_blocked
    best_score = 0.0
    best_set = []
    
    def dfs(avail_mask, curr_small_sum, blocked_large_mask, sum_blocked, curr_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        # Upper Bound
        # Max possible score = curr_small_sum + (total_large_sum - sum_blocked) + sum(1/x for x in avail)
        # This is because adding x increases curr_small_sum but might increase sum_blocked.
        # So ignoring the increase in sum_blocked gives an upper bound.
        
        # Calculate sum of 1/x for available
        # To make this fast, we can precompute a list of (mask, sum) for small masks? No.
        # Just loop.
        add_sum = 0.0
        for i in range(n):
            if avail_mask & (1 << i):
                add_sum += 1.0 / smalls[i]
                
        bound = curr_small_sum + (total_large_sum - sum_blocked) + add_sum
        
        if bound <= best_score:
            return
            
        if avail_mask == 0:
            score = curr_small_sum + (total_large_sum - sum_blocked)
            if score > best_score:
                best_score = score
                best_set = list(curr_set)
            return
            
        # Branch on a vertex
        # Heuristic: pick vertex with highest 1/x
        branch_idx = -1
        max_val = -1
        for i in range(n):
            if avail_mask & (1 << i):
                val = 1.0 / smalls[i]
                if val > max_val:
                    max_val = val
                    branch_idx = i
                    
        if branch_idx == -1:
            return
            
        x = smalls[branch_idx]
        
        # 1. Include x
        new_avail = avail_mask & ~(adj[branch_idx] | (1 << branch_idx))
        new_curr_small_sum = curr_small_sum + 1.0 / x
        new_blocked_mask = blocked_large_mask | blocked_mask_large[branch_idx]
        
        # Calculate new sum_blocked
        # delta = sum(1/y for y in (new_blocked_mask & ~blocked_large_mask))
        # We can precompute an array sum_1_over_large[y]
        # And iterate over bits in delta_mask
        delta_mask = new_blocked_mask & ~blocked_large_mask
        delta_sum = 0.0
        for k in range(128):
            if delta_mask & (1 << k):
                delta_sum += 1.0 / (129 + k)
                
        new_sum_blocked = sum_blocked + delta_sum
        new_set = curr_set + [x]
        
        dfs(new_avail, new_curr_small_sum, new_blocked_mask, new_sum_blocked, new_set)
        
        # 2. Exclude x
        new_avail_excl = avail_mask & ~(1 << branch_idx)
        dfs(new_avail_excl, curr_small_sum, blocked_large_mask, sum_blocked, curr_set)

    # Start DFS
    dfs((1 << n) - 1, 0.0, 0, 0.0, [])
    
    # Construct final set
    S_small = best_set
    S_large = []
    blocked_mask_final = 0
    for x in S_small:
        idx = smalls.index(x)
        blocked_mask_final |= blocked_mask_large[idx]
        
    for k, y in enumerate(range(129, 257)):
        if not (blocked_mask_final & (1 << k)):
            S_large.append(y)
            
    return S_small + S_large
