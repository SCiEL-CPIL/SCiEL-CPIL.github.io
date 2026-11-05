<<<<<<< SEARCH
    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 4. Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)
=======
    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 3b. 2-swap local search
    if best_blocks and time.time() < deadline - 5.0:
        improved_blocks = local_search_2swap(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 4. Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)
>>>>>>> REPLACE

