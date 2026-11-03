<<<<<<< SEARCH
    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with primal-dual greedy
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        rng = random.Random(seed * 7919 + 13)
        blocks = _primal_dual_greedy(rng, deadline - 30)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    if best_blocks and time.time() < deadline - 10:
        best_blocks = _local_search(best_blocks, deadline - 5)
        if len(best_blocks) < best_len:
            best_len = len(best_blocks)

    if best_blocks and time.time() < deadline - 5:
        best_blocks = _two_block_swap(best_blocks, deadline)
        if len(best_blocks) < best_len:
            best_len = len(best_blocks)

    return best_blocks if best_blocks else []
=======
    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with primal-dual greedy (now with randomization)
    n_restarts = 15
    for seed in range(n_restarts):
        if time.time() > deadline - 25:
            break
        rng = random.Random(seed * 7919 + 13)
        blocks = _primal_dual_greedy(rng, deadline - 25)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # Local search on best solution
    if best_blocks and time.time() < deadline - 15:
        refined = _local_search(best_blocks, deadline - 10)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(best_blocks)

    # Two-block swap refinement
    if best_blocks and time.time() < deadline - 5:
        refined = _two_block_swap(best_blocks, deadline - 1)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(best_blocks)

    return best_blocks if best_blocks else []
>>>>>>> REPLACE
