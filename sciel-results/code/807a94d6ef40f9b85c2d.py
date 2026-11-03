<<<<<<< SEARCH
    # Local search polish
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2:
        p_final, cost_final, _ = _local_search(
            a, b, p_final, cost_final,
            max_iter=5000,
            time_limit=remaining_time - 1,
            start_time=start_time
        )
=======
    # Local search polish
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2:
        p_final, cost_final, _ = _surrogate_local_search(
            a, b, p_final, cost_final,
            max_iter=5000,
            time_limit=remaining_time - 1,
            start_time=start_time
        )
>>>>>>> REPLACE
