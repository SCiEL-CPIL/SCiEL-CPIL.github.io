def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best = None
    best_size = N + 1
    
    candidates = []
    
    # 1. Greedy Ascending
    c1 = _greedy_bitset(list(range(1, N + 1)), deadline)
    candidates.append(c1)
    
    # 2. Greedy Descending
    c2 = _greedy_bitset(list(range(N, 0, -1)), deadline)
    candidates.append(c2)
    
    # 3. Randomized Greedy - Increased count
    for seed in range(150):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng = random.Random(1000 + seed)
        rng.shuffle(order)
        sel = _greedy_bitset(order, deadline)
        if sel:
            candidates.append(sel)
            
    # 4. Targeted Randomized Greedy
    for seed in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng = random.Random(2000 + seed)
        rng.shuffle(order)
        sel = _greedy_bitset(order, deadline)
        if sel:
            candidates.append(sel)

    # Process candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if not cand:
            continue
        
        # Ensure valid Sidon
        if not _is_sidon(cand):
            continue
            
        # Check if maximal.
        sums = _compute_sums_bitset(cand)
        if not _is_maximal_bitset(cand, sums):
            continue
            
        rng = random.Random(hash(tuple(sorted(cand))) % (2 ** 31))
        
        # Reduce
        minimized = _reduce_bitset(cand, deadline, rng)
        
        # Swap Improve
        if time.time() < deadline - 5.0:
            minimized = _swap_improve_bitset(minimized, deadline, rng)
            # Reduce again after swaps
            minimized = _reduce_bitset(minimized, deadline, rng)
            
        if minimized:
            if _is_sidon(minimized) and _is_maximal_bitset(minimized, _compute_sums_bitset(minimized)):
                if len(minimized) < best_size:
                    best_size = len(minimized)
                    best = list(minimized)

    if best is None:
        # Fallback
        return _greedy_bitset(list(range(1, N + 1)), start_time + 1.0)
        
    return best
