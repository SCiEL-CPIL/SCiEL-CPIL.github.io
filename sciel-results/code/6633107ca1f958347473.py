<<<<<<< SEARCH
def _simulated_annealing(all_points, start_set, max_iters=20000):
    """
    Simulated annealing to minimize the size of a maximal no-isosceles set.
    Moves: remove a point (if result is still valid), then try to add back greedily.
    Or swap operations.
    """
    current = list(start_set)
    current_set = set(current)
    best = list(current)
    best_len = len(current)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
        
        if T < T_min:
            break
        
        # Random move: try to remove a random point
        if current:
            idx = rng.randint(0, len(current) - 1)
            p = current[idx]
            new_set = list(current)
            new_set.pop(idx)
            new_set_list = [x for x in new_set]
            
            # Check if still valid (no isosceles)
            valid = True
            for i in range(len(new_set_list)):
                for j in range(i + 1, len(new_set_list)):
                    for k in range(j + 1, len(new_set_list)):
                        if _is_iso(new_set_list[i], new_set_list[j], new_set_list[k]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check if still maximal
                if _is_maximal(new_set_list, all_points):
                    # Accept (always accept if smaller)
                    current = new_set_list
                else:
                    # Not maximal - try to make it maximal by adding points
                    extended = _construct_maximal(new_set_list, all_points)
                    delta = len(extended) - len(current)
                    if delta <= 0 or rng.random() < (1.0 - delta) / T:
                        current = extended
            # else: invalid, skip
        
        T *= cooling
    
    return best
=======
def _simulated_annealing(all_points, start_set, max_iters=20000, deadline=None):
    """
    Simulated annealing to minimize the size of a maximal no-isosceles set.
    Moves: remove a point, then try to add back greedily.
    """
    current = list(start_set)
    best = list(current)
    best_len = len(current)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
        
        if T < T_min:
            break
        
        # Random move: try to remove a random point
        if current:
            idx = rng.randint(0, len(current) - 1)
            p = current[idx]
            new_set = list(current)
            new_set.pop(idx)
            
            # Removing a point always preserves validity (fewer triangles)
            # Make maximal
            extended = _construct_maximal(new_set, all_points)
            delta = len(extended) - len(current)
            if delta <= 0 or (T > 0 and rng.random() < (1.0 - delta) / T):
                current = extended
        
        T *= cooling
    
    return best
>>>>>>> REPLACE

