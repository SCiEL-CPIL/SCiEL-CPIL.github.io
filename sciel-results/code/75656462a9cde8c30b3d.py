<<<<<<< SEARCH
    # 3) Active-set refinement on the best layout
    if time.time() - start_time < DEADLINE - 10:
        as_budget = min(5.0, DEADLINE - 10 - (time.time() - start_time))
        if as_budget > 1.0:
            cur = _active_set_refine(best, best_r, time_budget=as_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
=======
    # 3) Combinatorial re-seeding: Try to find a new grid-based layout 
    # that might be in a different basin of attraction
    if time.time() - start_time < DEADLINE - 15:
        # Use a radius slightly below the current best to find a new valid config
        # If we find one, it might lead to a better continuous optimum
        target_r = best_r * 0.98
        c_new_grid = _combinatorial_grid_search(target_r, time_budget=3.0, rng=np.random.default_rng(123))
        if c_new_grid is not None:
            # Refine this new grid layout
            cur = _force_relax(c_new_grid, target_r, time_budget=2.0, max_iter=3000)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
            # Even if not immediately better, use it as a candidate for softmin
            # We'll add it to the candidates list later

    # 4) Active-set refinement on the best layout
    if time.time() - start_time < DEADLINE - 10:
        as_budget = min(5.0, DEADLINE - 10 - (time.time() - start_time))
        if as_budget > 1.0:
            cur = _active_set_refine(best, best_r, time_budget=as_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
>>>>>>> REPLACE

And update the candidate list for softmin to include the combinatorial result if it was found.

