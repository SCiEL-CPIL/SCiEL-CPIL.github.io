<<<<<<< SEARCH
    # Run Simulated Annealing on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        sa_result = simulated_annealing(best_blocks, deadline - 5.0)
        if len(sa_result) < best_len:
            best_blocks = sa_result
            best_len = len(best_blocks)
=======
    # Run Simulated Annealing on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        sa_result = simulated_annealing(best_blocks, deadline - 5.0)
        if len(sa_result) < best_len:
            best_blocks = sa_result
            best_len = len(best_blocks)
        
        # Try SA on a few more random solutions if time permits
        if time.time() < deadline - 10.0:
            for extra_seed in [7, 13, 23]:
                if time.time() >= deadline - 5.0:
                    break
                extra_idx = greedy_cover_min_degree(extra_seed)
                extra_idx = reduce_blocks(extra_idx)
                if extra_idx and len(extra_idx) < best_len:
                    best_blocks = extra_idx
                    best_len = len(extra_idx)
                sa_extra = simulated_annealing(extra_idx, deadline - 5.0)
                if len(sa_extra) < best_len:
                    best_blocks = sa_extra
                    best_len = len(best_blocks)
>>>>>>> REPLACE
