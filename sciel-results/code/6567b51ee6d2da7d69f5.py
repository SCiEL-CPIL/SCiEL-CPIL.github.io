<<<<<<< SEARCH
    # Strategy 3+: Random restarts with weighted greedy + simulated annealing
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        use_weighted = (r % 2 == 0)
        sel = greedy_cover(use_weighted=use_weighted, random_tiebreak=True, rng=rng)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply simulated annealing
            if time.time() < deadline - 15.0:
                sa_result = simulated_annealing(peeled, rng, max_iters=2000)
                sa_size = len(sa_result)
                if sa_size < size:
                    size = sa_size
                    peeled = sa_result
            
            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]
=======
    # Strategy 3+: Random restarts with weighted greedy + simulated annealing
    num_restarts = 25
    for r in range(num_restarts):
        if time.time() > deadline - 35.0:
            break
        
        use_weighted = (r % 2 == 0)
        sel = greedy_cover(use_weighted=use_weighted, random_tiebreak=True, rng=rng)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply simulated annealing
            if time.time() < deadline - 20.0:
                sa_result = simulated_annealing(peeled, rng, max_iters=3000)
                sa_size = len(sa_result)
                if sa_size < size:
                    size = sa_size
                    peeled = sa_result
            
            if size < best_size:
                best_size = size
                best_solution = [list(all_blocks[i]) for i in peeled]
>>>>>>> REPLACE
