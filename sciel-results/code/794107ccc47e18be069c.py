<<<<<<< SEARCH
def _search_warnsdorff(start, time_limit, rng):
    """
    Perform a search using Warnsdorff's rule with backtracking.
    Returns the best path found within time_limit.
    """
    deadline = time.time() + time_limit
    best_path = [start]
    best_len = 1
    
    # Stack for backtracking: stores (path, used, index_of_last_decision)
    # We use a simpler recursive-like structure with manual stack to avoid recursion limits
    # Actually, for Python, a depth-first search with backtracking is standard.
    
    path = [start]
    used = {start}
    
    # We will use a list to store the history of choices for backtracking
    # history[i] is the list of available neighbors at step i, and which one was chosen
    history = []
    
    def backtrack(depth):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return False
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        if depth == len(path) - 1:
            cur = path[-1]
            
            # Generate candidate moves
            candidates = []
            for nb in _NEIGHBORS[cur]:
                if nb in used:
                    continue
                # Check safety
                safe = True
                for p in path[:-1]:
                    if _BIT_COUNT[nb ^ p] == 1:
                        safe = False
                        break
                if safe:
                    # Calculate Warnsdorff score
                    score = _count_safe_moves(nb, path, used)
                    candidates.append((score, rng.random(), nb))
            
            if not candidates:
                return False # Dead end, need to backtrack
            
            # Sort by score (ascending), then random tie-breaker
            candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Record history for backtracking
            # We store the list of candidates we considered, in order
            # We will try them one by one
            history.append(candidates)
            
            # Try the best candidate first
            for idx, (score, rand, nb) in enumerate(candidates):
                if time.time() > deadline:
                    return False
                
                # Make the move
                path.append(nb)
                used.add(nb)
                
                # Recurse
                if backtrack(depth + 1):
                    return True # Success or timeout, unwind
                
                # Undo move
                path.pop()
                used.discard(nb)
                
                # If we have tried all candidates and none worked, we are stuck
                # The caller will handle backtracking further up
            
            # If we looped through all candidates and returned False from recursion,
            # it means all branches from this node are exhausted or timed out.
            # We remove the history entry as we are backtracking past this level.
            history.pop()
            return False
        
        return True

    # Start the search
    # To avoid deep recursion issues, we limit the depth or rely on the time limit
    # Q_9 has 512 vertices, so max depth is 512. Python recursion limit is usually 1000, so it's safe.
    backtrack(0)
    
    return best_path
=======
def _search_warnsdorff(start, time_limit, rng):
    """
    Perform a search using Warnsdorff's rule with backtracking.
    Returns the best path found within time_limit.
    """
    deadline = time.time() + time_limit
    best_path = [start]
    best_len = 1
    
    path = [start]
    used = {start}
    
    history = []
    
    def backtrack(depth):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return False
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        if depth == len(path) - 1:
            cur = path[-1]
            
            # Generate candidate moves
            candidates = []
            for nb in _NEIGHBORS[cur]:
                if nb in used:
                    continue
                # Check safety
                safe = True
                for p in path[:-1]:
                    if nb in _NBR_SETS[p]:
                        safe = False
                        break
                if safe:
                    # Calculate Warnsdorff score
                    score = _count_safe_moves(nb, path, used)
                    candidates.append((score, rng.random(), nb))
            
            if not candidates:
                return False # Dead end, need to backtrack
            
            # Sort by score (ascending), then random tie-breaker
            candidates.sort(key=lambda x: (x[0], x[1]))
            
            history.append(candidates)
            
            # Try the best candidate first
            for idx, (score, rand, nb) in enumerate(candidates):
                if time.time() > deadline:
                    return False
                
                # Make the move
                path.append(nb)
                used.add(nb)
                
                # Recurse
                if backtrack(depth + 1):
                    return True # Success or timeout, unwind
                
                # Undo move
                path.pop()
                used.discard(nb)
                
                # If we have tried all candidates and none worked, we are stuck
                # The caller will handle backtracking further up
            
            # If we looped through all candidates and returned False from recursion,
            # it means all branches from this node are exhausted or timed out.
            # We remove the history entry as we are backtracking past this level.
            history.pop()
            return False
        
        return True

    # Start the search
    backtrack(0)
    
    return best_path
>>>>>>> REPLACE

