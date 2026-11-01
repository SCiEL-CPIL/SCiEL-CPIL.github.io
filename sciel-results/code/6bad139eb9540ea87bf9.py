<<<<<<< SEARCH
    # Phase 4: Additional local search refinement
    current = list(best_blocks)
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False

        # Try local search removal (batch)
        reduced = _local_search_remove_redundant(current, v, k, t, all_tsub_set)
        if len(reduced) < len(current):
            current = reduced
            improved = True
            no_improve_count = 0
            continue

        # Try point swaps to enable deletion
        rng_ls = random.Random(int(time.time() * 1000000))
        new_blocks = _try_swap_reduce(current, v, k, t, rng_ls, all_tsub_set)
        if new_blocks is not None:
            current = new_blocks
            improved = True
            no_improve_count = 0
            continue

        no_improve_count += 1
=======
    # Phase 4: Additional local search refinement
    current = list(best_blocks)
    no_improve_count = 0
    max_no_improve = 50  # Increased iterations

    while time.time() < deadline - 2 and no_improve_count < max_no_improve:
        improved = False

        # Try local search removal (batch)
        reduced = _local_search_remove_redundant(current, v, k, t, all_tsub_set)
        if len(reduced) < len(current):
            current = reduced
            improved = True
            no_improve_count = 0
            continue

        # Try point swaps to enable deletion
        rng_ls = random.Random(int(time.time() * 1000000))
        new_blocks = _try_swap_reduce(current, v, k, t, rng_ls, all_tsub_set)
        if new_blocks is not None:
            current = new_blocks
            improved = True
            no_improve_count = 0
            continue

        no_improve_count += 1
>>>>>>> REPLACE
