<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # All candidates from 1 to 256
    all_candidates = list(range(1, 257))
    
    # Strategy: use larger numbers to reduce subset sum collisions
    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Multiple restarts with different starting points
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Randomly select a starting set of larger numbers
        large_candidates = [x for x in all_candidates if x >= 80]
        random.shuffle(large_candidates)
        
        # Start with 2-4 elements
        start_size = random.randint(2, 4)
        start_elements = large_candidates[:start_size]
        
        # Ensure start is valid
        if not _is_valid(start_elements):
            continue
        
        # Greedy add with sorted candidates (larger first to reduce collisions)
        remaining = [x for x in large_candidates if x not in set(start_elements)]
        random.shuffle(remaining)
        current = _greedy_construct(start_elements, remaining, deadline)
        
        # Local search with all remaining candidates
        all_remaining = [x for x in all_candidates if x not in set(current)]
        current = _local_search(current, all_remaining, deadline, max_iter=150)
        
        if len(current) > best_size:
            best_result = list(current)
            best_size = len(current)
        
        # Early termination if we hit a good size
        if best_size >= 12:
            break
    
    # Final verification
    if best_result and not _is_valid(best_result):
        # Fallback to a known small valid set
        best_result = [31, 32, 34]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # All candidates from 1 to 256
    all_candidates = list(range(1, 257))
    
    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Shuffle candidates
        shuffled_candidates = all_candidates[:]
        random.shuffle(shuffled_candidates)
        
        # Greedy construction
        current = []
        current_set = set()
        
        for c in shuffled_candidates:
            if time.time() > deadline:
                break
            if c in current_set:
                continue
            if _is_valid(current + [c]):
                current.append(c)
                current_set.add(c)
        
        # Local search
        remaining = [x for x in all_candidates if x not in current_set]
        current = _local_search(current, remaining, deadline, max_iter=50)
        
        if len(current) > best_size:
            best_result = list(current)
            best_size = len(current)
            
        if best_size >= 20:
            break

    # Final verification
    if best_result and not _is_valid(best_result):
        best_result = [2, 3]
        
    return [int(x) for x in best_result]
>>>>>>> REPLACE
