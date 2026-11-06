import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    best_seq = []
    best_len = 0

    # Helper to update best
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()

    # 1. Greedy Initialization
    # This provides a good initial lower bound for best_len, which helps pruning.
    used = 0
    suffix_mask = 0
    greedy_seq = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        # Check validity using bitmasks
        # 1. Is x already a used sum?
        if (used >> x) & 1:
            continue
        # 2. Does x + s collide with any used sum for any suffix sum s?
        # This is equivalent to: (used >> x) & suffix_mask != 0
        if (used >> x) & suffix_mask:
            continue
        
        # Valid: Add x
        greedy_seq.append(x)
        # Update used: add x and all x+s
        used |= (1 << x) | (suffix_mask << x)
        # Update suffix_mask: new suffixes are x and x+s
        suffix_mask = (1 << x) | (suffix_mask << x)

    update_best(greedy_seq)

    # 2. DFS with Backtracking
    # State: last_val, used, suffix_mask, seq
    # We start from scratch to find the global optimum, but the greedy result
    # sets a high bar for pruning.
    
    def dfs(last_val, used, suffix_mask, seq):
        nonlocal best_seq, best_len
        
        current_len = len(seq)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
        
        # Time check
        if time.time() > deadline:
            return
        
        # Base case / Pruning
        if last_val >= 256:
            return
        
        # Upper bound: max possible additional elements
        # We can add at most (256 - last_val) elements.
        if current_len + (256 - last_val) <= best_len:
            return
            
        # Iterate candidates
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            # Check validity
            if (used >> x) & 1:
                continue
            if (used >> x) & suffix_mask:
                continue
            
            # Apply change
            # new_used = used | (1 << x) | (suffix_mask << x)
            # new_suffix_mask = (1 << x) | (suffix_mask << x)
            
            seq.append(x)
            
            dfs(x, used | (1 << x) | (suffix_mask << x), 
                (1 << x) | (suffix_mask << x), seq)
            
            seq.pop()

    # Run DFS
    dfs(0, 0, 0, [])

    return best_seq
