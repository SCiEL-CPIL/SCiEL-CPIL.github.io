<<<<<<< SEARCH
    # Multiple restarts with different seeds and strategies
    attempts = 0
    strategies = ['random', 'bit_priority', 'parity_weighted', 'clique_breaking', 'complementary', 'complementary']
    
    while time_left() > 15:
        attempts += 1
        seed = attempts * 7919 + 42
        strategy = strategies[attempts % len(strategies)]
        
        if strategy == 'complementary':
            sol = complementary_deletion(seed)
            # Perturb complementary solution slightly to escape local optima
            if attempts % 2 == 0 and best_solution:
                # Mix with best solution? Or just perturb
                rng = random.Random(seed)
                # Remove a few random edges from the complementary solution
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(5, len(selected) // 10))
                    for ei in to_remove:
                        sol[ei] = False
        else:
            # Alternate between random greedy and perturbed deterministic
            if attempts % 5 == 1:
                sol = greedy_initial(seed, strategy)
            elif attempts % 5 == 2:
                sol = list(best_solution) if best_solution else greedy_initial(seed, strategy)
                # Perturb: remove a few random edges
                rng = random.Random(seed)
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(3, len(selected) // 10))
                    for ei in to_remove:
                        sol[ei] = False
            elif attempts % 5 == 3:
                sol = greedy_initial(seed + 12345, 'random')
            elif attempts % 5 == 4:
                sol = greedy_initial(seed + 99999, 'bit_priority')
            else:
                sol = greedy_initial(seed + 55555, 'parity_weighted')
        
        # Local search
        sol = local_search(sol, seed)
        
        # Additional swap search
        if time_left() > 30:
            sol = swap_search(sol, seed + 777)
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 30:
            break
=======
    # Multiple restarts with different seeds and strategies
    attempts = 0
    strategies = ['random', 'bit_priority', 'parity_weighted', 'clique_breaking', 'complementary', 'complementary']
    
    while time_left() > 15:
        attempts += 1
        seed = attempts * 7919 + 42
        strategy = strategies[attempts % len(strategies)]
        
        if strategy == 'complementary':
            sol = complementary_deletion(seed)
            # Perturb complementary solution slightly to escape local optima
            if attempts % 2 == 0 and best_solution:
                # Mix with best solution? Or just perturb
                rng = random.Random(seed)
                # Remove a few random edges from the complementary solution
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(5, len(selected) // 10))
                    for ei in to_remove:
                        sol[ei] = False
        else:
            # Alternate between random greedy and perturbed deterministic
            if attempts % 5 == 1:
                sol = greedy_initial(seed, strategy)
            elif attempts % 5 == 2:
                sol = list(best_solution) if best_solution else greedy_initial(seed, strategy)
                # Perturb: remove a few random edges
                rng = random.Random(seed)
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(3, len(selected) // 10))
                    for ei in to_remove:
                        sol[ei] = False
            elif attempts % 5 == 3:
                sol = greedy_initial(seed + 12345, 'random')
            elif attempts % 5 == 4:
                sol = greedy_initial(seed + 99999, 'bit_priority')
            else:
                sol = greedy_initial(seed + 55555, 'parity_weighted')
        
        # Local search
        sol = local_search(sol, seed)
        
        # Additional swap search
        if time_left() > 30:
            sol = swap_search(sol, seed + 777)
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 30:
            break
    
    # Final high-precision polish pass on the best solution
    if best_solution is not None and time_left() > 25:
        final_sol = list(best_solution)
        final_sol = local_search(final_sol, seed=999999, max_iter=100000)
        if time_left() > 15:
            final_sol = swap_search(final_sol, seed=888888, max_iter=20000)
        
        final_cnt = count_edges(final_sol)
        if is_valid(final_sol) and final_cnt > best_count:
            best_count = final_cnt
            best_solution = final_sol
>>>>>>> REPLACE

