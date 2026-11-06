<<<<<<< SEARCH
    # Run local search on each start
    for i, A in enumerate(starts):
        if not _is_valid(A):
            # Fix if invalid
            A = sorted(set(A))
            if len(A) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in A]
                A += rng.sample(available, 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
            A = sorted(A)
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 1:
            continue
        
        # Allocate time proportional to remaining budget
        # Give first few starts more time
        alloc = max(0.5, remaining_time / (len(starts) - i))
        
        A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.5 and best_A:
        A_ref, obj_ref = _local_search(best_A, remaining, rng)
        if obj_ref < best_obj:
            best_A = A_ref
            best_obj = obj_ref
=======
    # Run local search on each start
    for i, A in enumerate(starts):
        if not _is_valid(A):
            # Fix if invalid
            A = sorted(set(A))
            if len(A) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in A]
                A += rng.sample(available, 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
            A = sorted(A)
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 1:
            continue
        
        # Allocate time: give more to earlier starts, but ensure each gets reasonable time
        # First 3 starts get 20% each, rest split remaining
        if i < 3:
            alloc = remaining_time * 0.2
        else:
            alloc = remaining_time / (len(starts) - i + 1)
        alloc = max(1.0, min(alloc, 15.0))  # Cap at 15s per start
        
        A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.0 and best_A:
        # Do multiple refinement passes
        for _ in range(3):
            remaining = time_budget - (time.time() - start_time)
            if remaining < 1.0:
                break
            A_ref, obj_ref = _local_search(best_A, remaining / 2, rng)
            if obj_ref < best_obj:
                best_A = A_ref
                best_obj = obj_ref
>>>>>>> REPLACE
