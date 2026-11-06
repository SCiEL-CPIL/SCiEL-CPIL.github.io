<<<<<<< SEARCH
def _remove_redundant_comparators(network, time_limit):
    """
    Try to remove each comparator if the network remains valid without it.
    Iterates repeatedly until no more can be removed or time runs out.
    """
    start_time = time.time()
    current = list(network)
    changed = True
    
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        n = len(current)
        for idx in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            # Try removing comparator at idx
            # Optimization: Skip if we can't afford the check
            candidate = current[:idx] + current[idx + 1:]
            if _is_valid(candidate):
                current = candidate
                changed = True
                break # Restart loop from beginning as indices shifted
    return current
=======
def _remove_redundant_comparators(network, time_limit):
    """
    Aggressively try to remove comparators. Uses multiple strategies:
    1. Sequential deletion from front
    2. Sequential deletion from back  
    3. Random deletion attempts
    Iterates until no more can be removed or time runs out.
    """
    start_time = time.time()
    current = list(network)
    n_comparators = len(current)
    
    # Strategy 1: Sequential deletion from front
    while (time.time() - start_time) < time_limit:
        changed = False
        n = len(current)
        for idx in range(n):
            if (time.time() - start_time) >= time_limit:
                return current
            candidate = current[:idx] + current[idx + 1:]
            if _is_valid(candidate):
                current = candidate
                changed = True
                break  # Restart from beginning
        
        if not changed:
            break
    
    # Strategy 2: Sequential deletion from back
    while (time.time() - start_time) < time_limit:
        changed = False
        n = len(current)
        for idx in range(n - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                return current
            candidate = current[:idx] + current[idx + 1:]
            if _is_valid(candidate):
                current = candidate
                changed = True
                break
        
        if not changed:
            break
    
    # Strategy 3: Random deletion attempts to escape local minima
    max_random_attempts = 100
    attempts = 0
    while (time.time() - start_time) < time_limit and attempts < max_random_attempts:
        attempts += 1
        n = len(current)
        if n == 0:
            break
        idx = random.randint(0, n - 1)
        candidate = current[:idx] + current[idx + 1:]
        if _is_valid(candidate):
            current = candidate
            # Reset attempts counter since we made progress
            attempts = 0
    
    return current
>>>>>>> REPLACE

