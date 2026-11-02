<<<<<<< SEARCH
            # Try each valid candidate to see if it maintains coverage
            best_candidate = None
            best_unique_count = len(unique_ts) + 1  # minimum possible is len(unique_ts)
            
            for new_block_idx in valid_candidates[:50]:  # limit to 50 candidates for speed
                if time.time() > deadline - 3.0:
                    break
                new_ts_arr = block_to_ts_np[new_block_idx]
                
                # Check if replacing block i with new_block_idx maintains coverage
                # Temporarily remove block i's contribution
                test_cover = cover_count.copy()
                np.subtract.at(test_cover, ts_arr_i, 1)
                np.add.at(test_cover, new_ts_arr, 1)
                
                if np.min(test_cover) >= 1:
                    # Count how many t-subsets are uniquely covered by the new block
                    new_unique = np.sum(test_cover[new_ts_arr] == 1)
                    if best_candidate is None or new_unique < best_unique_count:
                        best_unique_count = new_unique
                        best_candidate = new_block_idx
=======
            # Try each valid candidate to see if it maintains coverage
            best_candidate = None
            best_unique_count = len(unique_ts) + 1  # minimum possible is len(unique_ts)
            ts_arr_i_set = set(ts_arr_i.tolist())
            
            for new_block_idx in valid_candidates[:30]:  # limit to 30 candidates for speed
                if time.time() > deadline - 3.0:
                    break
                new_ts_arr = block_to_ts_np[new_block_idx]
                new_ts_set = set(new_ts_arr.tolist())
                
                # Check if all unique t-subsets in block i are covered by new block
                if not all(ts_idx in new_ts_set for ts_idx in unique_ts):
                    continue
                
                # Count how many t-subsets would be uniquely covered by the new block
                new_unique = 0
                for ts_idx in new_ts_arr:
                    old_c = cover_count[ts_idx]
                    new_c = old_c + 1 - (1 if ts_idx in ts_arr_i_set else 0)
                    if new_c == 1:
                        new_unique += 1
                
                if best_candidate is None or new_unique < best_unique_count:
                    best_unique_count = new_unique
                    best_candidate = new_block_idx
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Run multiple randomized greedy
    num_runs = 150
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break

        explore = 0.0 if i < 50 else (0.1 if i < 100 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Local search on best solution - run multiple times with different random seeds
    for ls_seed in range(5):
        if time.time() > deadline - 5.0:
            break
        random.seed(ls_seed)
        best_solution, _ = local_search(best_solution, max_iters=8000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Final local search
    best_solution, _ = local_search(best_solution, max_iters=3000)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    return best_solution
=======
    # Run multiple randomized greedy
    num_runs = 120
    for i in range(1, num_runs):
        if time.time() > deadline - 50.0:
            break

        explore = 0.0 if i < 40 else (0.1 if i < 80 else 0.15)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Local search on best solution - run multiple times with different random seeds
    for ls_seed in range(8):
        if time.time() > deadline - 5.0:
            break
        random.seed(ls_seed)
        best_solution, _ = local_search(best_solution, max_iters=10000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Final local search
    best_solution, _ = local_search(best_solution, max_iters=5000)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    return best_solution
>>>>>>> REPLACE
