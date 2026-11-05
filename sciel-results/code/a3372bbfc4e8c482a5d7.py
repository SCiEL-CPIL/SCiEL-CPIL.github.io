def construct():
    import time
    start_time = time.time()
    DEADLINE = 55  # Safety margin

    N = 180
    
    # Precompute all forbidden triples: (a, b, c) with a < b < c and a*(b+c) == b*c
    # conflict_pairs[x] = list of (u, v) such that {x, u, v} is a forbidden triple
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            num = a * b
            if num % diff != 0:
                continue
            c = num // diff
            if c > b and c <= N:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    # Use a boolean list for fast membership testing
    in_set = [False] * (N + 1)
    
    # Initialize with high range
    result = set(range(91, N + 1))
    for i in result:
        in_set[i] = True
        
    def can_add(x):
        """Check if x can be added to current set defined by in_set."""
        for (u, v) in conflict_pairs[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    # Greedy addition: try adding x from 90 down to 1
    for x in range(90, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        if can_add(x):
            in_set[x] = True
            result.add(x)
    
    # Local Search: Swaps
    # Strategy: Iterate through elements in result. Try to remove x.
    # Then try to add the best y not in result.
    # To speed up, we can just try adding any y that becomes valid.
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        # We iterate over a snapshot of the result to avoid modification during iteration issues
        # But we modify in_set, so we need to be careful.
        # We will try to find a beneficial swap: remove x, add y.
        # Net change is 0, but might allow further additions later?
        # Actually, just swapping doesn't increase size.
        # The goal is to MAXIMIZE cardinality.
        # The greedy phase adds elements. The local search should try to UNBLOCK elements.
        # If we remove x, we might be able to add y1 AND y2?
        # Or just y1? If we just swap x for y1, size is constant.
        # We need to find a move that increases size.
        # Move: Remove x, Add y. If y was blocked ONLY by x (and maybe others that are also removed?), 
        # it's a swap.
        # To increase size, we need to remove k elements and add k+1 elements.
        
        # Simpler approach: 
        # 1. Try to add any y not in set. (Already done in greedy, but maybe order matters?)
        #    Greedy went 90->1. Maybe 1->90 works better? 
        #    Let's try a second greedy pass with different order if time permits?
        #    No, let's stick to local search that increases size.
        
        # Check if any y not in set can be added right now.
        # (This catches cases where greedy order missed something, though unlikely if greedy was thorough)
        for y in range(1, N + 1):
            if not in_set[y]:
                if can_add(y):
                    in_set[y] = True
                    result.add(y)
                    improved = True
        
        if not improved:
            # Try 1-2 swaps: Remove 1, Add 2.
            # For each x in result:
            #   Remove x
            #   Try to add y1, y2 not in result
            #   If we can add 2, keep it.
            #   If we can add 1, it's a swap (no gain, but might enable future gains? 
            #   Usually we want to maximize size, so swaps are only useful if they lead to additions).
            #   Given time constraints, we will focus on "Remove 1, Add 2" directly.
            
            for x in list(result):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Remove x
                in_set[x] = False
                result.discard(x)
                
                added_count = 0
                added_elements = []
                
                # Try to add elements from 1 to N that are not in set
                # We want to add as many as possible.
                # Iterate through all y not in set
                for y in range(1, N + 1):
                    if in_set[y]:
                        continue
                    if can_add(y):
                        in_set[y] = True
                        result.add(y)
                        added_count += 1
                        added_elements.append(y)
                        if added_count >= 2:
                            break # Found a 1->2 move, net gain +1
                
                if added_count == 0:
                    # Restore x
                    in_set[x] = True
                    result.add(x)
                elif added_count == 1:
                    # Swap: removed x, added y. Size unchanged.
                    # This might help if the new set allows more additions later.
                    # However, it doesn't immediately increase score.
                    # We might accept it to escape local optima.
                    # But for safety and simplicity, let's only accept if it leads to growth?
                    # Actually, if we swap, we should check if we can add MORE.
                    # The loop above stopped after 2. If it added 1, it continued to check for 2?
                    # No, the loop continues until break or end.
                    # If added_count == 1, it means only 1 was added.
                    # We keep the swap (x removed, y added) because it's neutral.
                    # improved = True to trigger another pass?
                    # Let's mark improved to ensure we don't get stuck.
                    improved = True
                else:
                    # added_count >= 2, net gain >= 1
                    improved = True
                
    return list(result)
