<<<<<<< SEARCH
def construct():
    rng = random.Random(42)
    best_mask = 0
    best_len = TOTAL + 1

    # Phase 1: Multi-start random greedy
    result_mask = _multi_start_greedy(rng, n_starts=100)
    if result_mask != 0:
        best_mask = result_mask
        best_len = bin(result_mask).count('1')

    # Phase 2: Constrained-first greedy
    result_mask = _constrained_first_greedy(rng, n_starts=40)
    if result_mask != 0 and bin(result_mask).count('1') < best_len:
        best_mask = result_mask
        best_len = bin(result_mask).count('1')

    # Phase 3: Local search on best
    if best_mask != 0:
        improved_mask = _local_search(best_mask, rng, max_iters=100)
        improved_len = bin(improved_mask).count('1')
        if improved_len < best_len:
            best_mask = improved_mask
            best_len = improved_len

    # Phase 4: More random starts if time permits
    if time.time() < DEADLINE - 30:
        result_mask = _multi_start_greedy(rng, n_starts=200)
        if result_mask != 0 and bin(result_mask).count('1') < best_len:
            best_mask = result_mask
            best_len = bin(result_mask).count('1')
        if best_mask != 0:
            improved_mask = _local_search(best_mask, rng, max_iters=100)
            improved_len = bin(improved_mask).count('1')
            if improved_len < best_len:
                best_mask = improved_mask
                best_len = improved_len

    # Phase 5: Final local search
    if best_mask != 0 and time.time() < DEADLINE - 10:
        improved_mask = _local_search(best_mask, rng, max_iters=200)
        if bin(improved_mask).count('1') < best_len:
            best_mask = improved_mask

    # Convert back to list of (x, y) tuples
    if best_mask == 0:
        return []
    
    result = []
    for idx in _mask_to_list(best_mask):
        result.append(_idx_to_point(idx))
    return result
=======
def construct():
    rng = random.Random(42)
    best_mask = 0
    best_len = TOTAL + 1

    # Phase 1: Multi-start random greedy
    result_mask = _multi_start_greedy(rng, n_starts=80)
    if result_mask != 0:
        best_mask = result_mask
        best_len = bin(result_mask).count('1')

    # Phase 2: Constrained-first greedy
    result_mask = _constrained_first_greedy(rng, n_starts=40)
    if result_mask != 0 and bin(result_mask).count('1') < best_len:
        best_mask = result_mask
        best_len = bin(result_mask).count('1')

    # Phase 3: Local search on best
    if best_mask != 0:
        improved_mask = _local_search(best_mask, rng, max_iters=300)
        improved_len = bin(improved_mask).count('1')
        if improved_len < best_len:
            best_mask = improved_mask
            best_len = improved_len

    # Phase 4: Branch and bound
    if best_mask != 0 and time.time() < DEADLINE - 60:
        bnb_result = _branch_and_bound(rng, best_mask, max_nodes=50000)
        bnb_len = bin(bnb_result).count('1')
        if bnb_len < best_len:
            best_mask = bnb_result
            best_len = bnb_len

    # Phase 5: More random starts if time permits
    if time.time() < DEADLINE - 40:
        result_mask = _multi_start_greedy(rng, n_starts=150)
        if result_mask != 0 and bin(result_mask).count('1') < best_len:
            best_mask = result_mask
            best_len = bin(result_mask).count('1')
        if best_mask != 0:
            improved_mask = _local_search(best_mask, rng, max_iters=200)
            improved_len = bin(improved_mask).count('1')
            if improved_len < best_len:
                best_mask = improved_mask
                best_len = improved_len

    # Phase 6: Final local search
    if best_mask != 0 and time.time() < DEADLINE - 15:
        improved_mask = _local_search(best_mask, rng, max_iters=300)
        if bin(improved_mask).count('1') < best_len:
            best_mask = improved_mask

    # Convert back to list of (x, y) tuples
    if best_mask == 0:
        return []
    
    result = []
    for idx in _mask_to_list(best_mask):
        result.append(_idx_to_point(idx))
    return result
>>>>>>> REPLACE
