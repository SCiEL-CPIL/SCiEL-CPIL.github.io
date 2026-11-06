<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin within 90s limit
    
    # Try to find a good cycle
    result = _find_best_cycle(deadline)
    
    # Validate: must be a permutation of 0..23
    if result is None or len(result) != N or set(result) != set(range(N)):
        result = list(range(N))
    
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin within 90s limit
    
    # Try to find a good cycle using binary search + backtracking
    result = _find_best_cycle(deadline)
    
    # Validate: must be a permutation of 0..23
    if result is None or len(result) != N or set(result) != set(range(N)):
        result = list(range(N))
    
    # Apply k-opt local search to further minimize max edge arrival
    if time.time() < deadline - 5:
        improved = _kopt_local_search(result, deadline)
        if len(improved) == N and set(improved) == set(range(N)):
            result = improved
    
    return result
>>>>>>> REPLACE
