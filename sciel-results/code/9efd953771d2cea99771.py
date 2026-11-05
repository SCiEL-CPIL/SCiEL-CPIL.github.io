<<<<<<< SEARCH
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        # Use a random greedy that we know works for small n
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
=======
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        # Use a random greedy that we know works for small n
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    else:
        # Try to push n higher with remaining time
        if _remaining_time() > 10:
            current_n = best_n
            for n_try in range(current_n + 1, 41):
                if _remaining_time() < 5:
                    break
                for seed in range(20):
                    if _remaining_time() < 3:
                        break
                    result = _try_construction_bitmask(n_try, seed + 10000)
                    if result is not None:
                        adj_res, alpha_res = result
                        if alpha_res < 10:
                            best_adj = adj_res
                            best_n = n_try
                            best_alpha = alpha_res
                            break
                if best_n > current_n:
                    current_n = best_n
>>>>>>> REPLACE
