<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    best_solution = None
    
    # Try progressively larger upper bounds
    # Start with small M to find minimal diameter solutions
    for M in range(35, 200):
        if time.time() > deadline - 10:
            break
            
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 0.5:
            break
            
        # Allocate more time to smaller M values where solutions are more likely
        if M < 50:
            m_deadline = min(deadline - 10, time.time() + 8.0)
        elif M < 80:
            m_deadline = min(deadline - 10, time.time() + 5.0)
        else:
            m_deadline = min(deadline - 10, time.time() + 3.0)
        
        result = _search_with_bitmask(M, target_diffs, m_deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                best_solution = result
                break
    
    # If we found a solution, try to polish it
    if best_solution is not None:
        polish_deadline = min(deadline, time.time() + 15.0)
        best_solution = _local_polish(best_solution, target_diffs, polish_deadline)
    
    # If no solution found, return fallback
    if best_solution is None:
        best_solution = _construct_fallback()
    
    return best_solution
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # Start with a known valid solution as baseline
    best_solution = _construct_fallback()
    best_max = max(best_solution)
    
    # Try to find solutions with smaller diameter
    # Start with small M to find minimal diameter solutions
    for M in range(35, best_max):
        if time.time() > deadline - 10:
            break
            
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 0.5:
            break
            
        # Allocate more time to smaller M values where solutions are more likely
        if M < 40:
            m_deadline = min(deadline - 10, time.time() + 10.0)
        elif M < 50:
            m_deadline = min(deadline - 10, time.time() + 5.0)
        else:
            m_deadline = min(deadline - 10, time.time() + 2.0)
        
        result = _search_with_bitmask(M, target_diffs, m_deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                if max(result) < best_max:
                    best_solution = result
                    best_max = max(result)
                break
    
    # Try to polish the best solution
    if best_solution is not None:
        polish_deadline = min(deadline, time.time() + 20.0)
        polished = _local_polish(best_solution, target_diffs, polish_deadline)
        if polished is not None and _check_sidon_and_cover(polished, target_diffs):
            if max(polished) <= best_max:
                best_solution = polished
    
    return best_solution
>>>>>>> REPLACE
