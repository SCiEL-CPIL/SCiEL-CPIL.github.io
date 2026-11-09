def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 17
    MIN_D = 6
    MAX_SPACE = 1 << N
    
    # Precompute weights for all possible words
    weights = [0] * MAX_SPACE
    for i in range(1, MAX_SPACE):
        weights[i] = weights[i >> 1] + (i & 1)
    
    # Greedy basis construction for a linear code
    # We try to find a basis of size k such that all combinations have weight >= MIN_D
    # This gives a code of size 2^k
    
    basis = []
    current_code = [0]  # List of all combinations of current basis
    
    # We want to find 8 rows (size 256)
    # We scan candidates in a fixed order.
    
    for g in range(1, MAX_SPACE):
        if time.time() > deadline:
            break
            
        # Check if g can be added to the basis
        # Condition: wt(g + c) >= MIN_D for all c in current_code
        valid = True
        for c in current_code:
            if weights[g ^ c] < MIN_D:
                valid = False
                break
        
        if valid:
            basis.append(g)
            # Update current_code
            # New code = old code union (old code + g)
            new_code = []
            for c in current_code:
                new_code.append(c)
                new_code.append(c ^ g)
            current_code = new_code
            
            if len(basis) >= 8:
                break
                
    # If we have time, try to add non-linear words to the linear code
    # Use a forbidden array to speed up checks
    
    # Build forbidden array from current code
    forbidden = [False] * MAX_SPACE
    for w in current_code:
        forbidden[w] = True
    
    # Precompute masks with weight <= 5
    # These are the vectors within distance 5
    masks = []
    for m in range(MAX_SPACE):
        if weights[m] <= 5:
            masks.append(m)
    
    # Greedy fill: try to add more words
    code_set = set(current_code)
    
    while time.time() < deadline:
        added_any = False
        # Iterate through all possible words
        for w in range(MAX_SPACE):
            if not forbidden[w]:
                # Check if w can be added
                # We need to check distance to all existing words
                # But we can use the forbidden array: if w is not forbidden,
                # it means no existing word is within distance 5
                # So we can just add it
                # Wait, the forbidden array marks words that are within distance 5 of existing words
                # So if forbidden[w] is False, w is at distance >= 6 from all existing words
                # So we can add it
                code_set.add(w)
                current_code.append(w)
                # Mark w and its neighbors as forbidden
                for m in masks:
                    idx = w ^ m
                    forbidden[idx] = True
                added_any = True
            
            # Check time periodically to avoid exceeding deadline mid-loop
            if (w & 0xFF) == 0 and time.time() > deadline:
                break
        
        if not added_any:
            # No more words can be added with this greedy strategy
            break
            
    return current_code
