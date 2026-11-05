<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 30.0:
        bt_rows, bt_scope = _randomized_backtracking(n_rows, k, deadline - 30.0, rng, initial_max=best_scope)
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Final refinement with difference-swap mutations
    if time.time() < deadline - 2.0:
        used = _all_diffs(best_rows)
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng)
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows
=======
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 125.0  # Tighter safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Randomized backtracking with decreasing bounds
    # Try multiple starting max values to find good initial solutions
    if time.time() < deadline - 20.0:
        bt_rows, bt_scope = _randomized_backtracking(n_rows, k, deadline - 20.0, rng, initial_max=best_scope)
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Final refinement with difference-swap mutations
    if time.time() < deadline - 2.0:
        used = _all_diffs(best_rows)
        num_iterations = 0
        while time.time() < deadline - 2.0 and num_iterations < 20:
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=rng.randint(1, 3), rng=rng)
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
            num_iterations += 1
    
    return best_rows
>>>>>>> REPLACE

