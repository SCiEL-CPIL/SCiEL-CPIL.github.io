<<<<<<< SEARCH
def local_search(A, time_limit, rng, candidate_pool):
    """
    Perform greedy local search to minimize the score.
    """
    A = list(A)
    # Sort to keep consistency, though order doesn't affect set sizes
    A.sort()
    
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try replacing each element
        # We iterate backwards in case we remove/insert, but we just swap values here
        # We don't change the size of A, just the values.
        
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            # Try candidates
            # Optimization: Skip candidates that are too close to existing elements? No, let's try all.
            # However, we can pre-filter candidates that are already in A.
            A_set = set(A)
            
            for new_val in candidate_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                # Create new A
                # Instead of copying the whole list, we can modify and revert, 
                # but calc_score needs the list. 
                # For n=20, copying is cheap.
                new_A = A.copy()
                new_A[i] = new_val
                # Keep sorted for consistency (optional, but good for debugging)
                # new_A.sort() -- actually, calc_score doesn't require sorted.
                
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    # Restart the inner loop? Or continue?
                    # If we change A[i], the set of valid candidates changes slightly (old_val is now free, new_val is taken).
                    # We break out of the candidate loop for this i to re-evaluate the state.
                    # Actually, we can just break and let the outer while loop handle the next pass.
                    # But continuing to the next i in the same pass is also standard.
                    # Let's break to update the A_set reference implicitly? 
                    # No, A_set is a local snapshot. 
                    # We should break the candidate loop to ensure we don't use stale info or just proceed.
                    # Proceeding to next i is fine.
                    pass 
            # We don't break the i-loop, we continue to check other positions
            # This allows multiple changes per "pass"
            
    return best_A, best_score
=======
def local_search(A, time_limit, rng, candidate_pool):
    """
    Perform greedy local search with simulated annealing to escape local minima.
    """
    A = list(A)
    A.sort()
    
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Phase 1: Greedy local search
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.4:
        improved = False
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit * 0.4:
                break
            
            old_val = A[i]
            A_set = set(A)
            
            for new_val in candidate_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break  # Restart scanning from this position
    
    # Phase 2: Simulated Annealing to escape local minima
    if (time.time() - start_time) < time_limit * 0.85:
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling = 0.9995
        
        sa_start = time.time()
        sa_deadline = start_time + time_limit * 0.85
        
        while T > T_min and (time.time() - sa_start) < (time_limit * 0.45):
            # Random move: pick a position and a random candidate
            i = rng.randint(0, len(A) - 1)
            old_val = A[i]
            A_set = set(A)
            
            # Try a few random candidates
            found_move = False
            for _ in range(10):
                new_val = rng.choice(candidate_pool)
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                new_score = calc_score(new_A)
                
                delta = new_score - current_score
                if delta < 0:
                    # Accept improvement
                    A = new_A
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(A)
                else:
                    # Accept with probability exp(-delta/T)
                    if rng.random() < math.exp(-delta / T):
                        A = new_A
                        current_score = new_score
                
                found_move = True
                break
            
            T *= cooling
    
    return best_A, best_score
>>>>>>> REPLACE

