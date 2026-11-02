<<<<<<< SEARCH
    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.50  # Leave more time for final local search and SA

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace
        ls_time = min(2.0, max(0.1, (time_limit - (time.time() - start_time)) * 0.1))
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 10:
            refined = local_search_replace(refined, min(1.5, max(0.1, (time_limit - (time.time() - start_time)) * 0.05)))
        if time.time() - start_time < time_limit - 8:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:  # Good enough, stop early
            break

    # Final polish on the best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polish_time = min(10.0, max(0.5, time_limit - (time.time() - start_time) - 3.0))
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 3:
            polished = local_search_replace(polished, min(5.0, max(0.5, time_limit - (time.time() - start_time)) - 3.0))
        if time.time() - start_time < time_limit - 2:
            polished = local_search_remove(polished)
        
        # Add simulated annealing phase
        if time.time() - start_time < time_limit - 1:
            sa_time = min(8.0, max(0.5, time_limit - (time.time() - start_time) - 1.0))
            sa_rng = random.Random(12345)
            sa_result = simulated_annealing(polished, sa_time, sa_rng)
            if len(sa_result) < len(polished):
                polished = sa_result
        
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 0.5:
            polished = local_search_single(polished, min(2.0, max(0.2, time_limit - (time.time() - start_time)) - 0.5)))
        best_blocks = polished
=======
    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.45  # Leave time for final local search

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove, then 2-for-1 replace, then 2-for-2 exchange
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 12:
            refined = local_search_replace(refined, min(1.5, max(0.1, (time_limit - (time.time() - start_time)) * 0.04)))
        if time.time() - start_time < time_limit - 10:
            refined = local_search_remove(refined)
        if time.time() - start_time < time_limit - 8:
            refined = local_search_exchange2(refined, min(1.0, max(0.1, (time_limit - (time.time() - start_time)) * 0.03)))
        if time.time() - start_time < time_limit - 7:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:  # Good enough, stop early
            break

    # Final polish on the best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 4:
            polished = local_search_replace(polished, min(4.0, max(0.5, time_limit - (time.time() - start_time)) - 4.0))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 2:
            polished = local_search_exchange2(polished, min(5.0, max(0.5, time_limit - (time.time() - start_time)) - 2.0))
        if time.time() - start_time < time_limit - 1.5:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 1:
            polished = local_search_single(polished, min(2.0, max(0.2, time_limit - (time.time() - start_time)) - 1.0))
        if time.time() - start_time < time_limit - 0.5:
            polished = local_search_remove(polished)
        best_blocks = polished
>>>>>>> REPLACE
