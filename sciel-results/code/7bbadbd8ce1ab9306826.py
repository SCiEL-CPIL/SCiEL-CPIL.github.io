<<<<<<< SEARCH
def construct():
    deadline = time.time() + 80  # Safety margin
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 257))
    
    orderings = []
    orderings.append(list(range(1, 257)))
    orderings.append(list(range(256, 0, -1)))
    orderings.append([x for x in range(1, 257, 2)] + [x for x in range(2, 257, 2)])
    orderings.append([x for x in range(2, 257, 2)] + [x for x in range(1, 257, 2)])
    alt = []
    for i in range(128):
        alt.append(1 + i)
        alt.append(129 + i)
    orderings.append(alt)
    
    # Add more structured orderings
    for start in range(1, 257, 16):
        o = []
        for i in range(128):
            v1 = start + i
            v2 = start + 128 + i
            if v1 <= 256:
                o.append(v1)
            if v2 <= 256:
                o.append(v2)
        orderings.append(o)
    
    # Add random orderings
    num_random = 30
    for _ in range(num_random):
        o = candidates[:]
        random.shuffle(o)
        orderings.append(o)
    
    # Greedy phase
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        current_set = _grow_greedy(ordering, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Local search phase with remaining time
    if best_size > 0 and time.time() < deadline - 5:
        improved_set = _swap_search(best_set[:], deadline - 2)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return best_set
=======
def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 257))
    
    orderings = []
    # Key orderings
    orderings.append(list(range(256, 0, -1)))  # High to low
    orderings.append(list(range(1, 257)))  # Low to high
    orderings.append([x for x in range(256, 0, -2)] + [x for x in range(255, 0, -2)])  # Odd high, even high
    orderings.append([x for x in range(255, 0, -2)] + [x for x in range(256, 0, -2)])  # Even high, odd high
    
    # Interleaved patterns
    for offset in [0, 1, 2, 3, 4, 5, 6, 7]:
        o = []
        for i in range(128):
            v1 = 256 - i
            v2 = 128 + offset + i
            if v1 >= 1:
                o.append(v1)
            if 1 <= v2 <= 256 and v2 != v1:
                o.append(v2)
        orderings.append(o)
    
    # Random orderings with bias toward larger numbers
    num_random = 40
    for _ in range(num_random):
        o = candidates[:]
        # Weight toward larger numbers
        weighted = []
        for x in o:
            weight = x // 10 + 1
            weighted.extend([x] * weight)
        random.shuffle(weighted)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for x in weighted:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        orderings.append(deduped)
    
    # Pure random
    for _ in range(20):
        o = candidates[:]
        random.shuffle(o)
        orderings.append(o)
    
    # Greedy phase
    for ordering in orderings:
        if time.time() > deadline - 10:  # Reserve time for local search
            break
        
        current_set = _grow_greedy(ordering, deadline - 5)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # If we get a good result, try backtracking
        if len(current_set) > 30 and len(current_set) < 60 and time.time() < deadline - 15:
            bt_set = _grow_greedy_with_backtrack(ordering, deadline - 10, max_depth=2)
            if len(bt_set) > best_size:
                best_size = len(bt_set)
                best_set = bt_set[:]
    
    # Local search phase with remaining time
    if best_size > 0 and time.time() < deadline - 3:
        improved_set = _swap_search(best_set[:], deadline - 1)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return best_set
>>>>>>> REPLACE

Let me verify the changes make sense and add one more improvement - a final verification and cleanup:

