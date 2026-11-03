<<<<<<< SEARCH
def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Uses efficient incremental diff counting.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000
    
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count(list(diff_counter.elements()))
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    # Cost function
    collision_weight = 100000
    
    def cost(scope, coll):
        return coll * collision_weight + scope
    
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        
        # Choose random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 10000
        
        # Move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to reduce (especially if near max)
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            # Larger move
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue
        
        # Compute old diffs for this row
        old_row = current[i]
        old_diffs = _diffs_for_row(old_row)
        
        # Apply change
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        
        # Check internal validity (all diffs positive and distinct within row)
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue
        
        # Update diff counter
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        
        new_collisions = _collision_count(list(diff_counter.elements()))
        
        # New scope
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < old_val:
            # Need to recompute
            new_scope = _scope(current)
        
        old_c = cost(current_scope, collisions)
        new_c = cost(new_scope, new_collisions)
        
        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob
        
        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best
=======
def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Uses efficient incremental diff counting.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000
    
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count(list(diff_counter.elements()))
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    # Cost function
    collision_weight = 100000
    
    def cost(scope, coll):
        return coll * collision_weight + scope
    
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        
        # Choose random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 10000
        
        # Move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to reduce (especially if near max)
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            # Larger move
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue
        
        # Compute old diffs for this row
        old_row = current[i]
        old_diffs = _diffs_for_row(old_row)
        
        # Apply change
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        
        # Check internal validity (all diffs positive and distinct within row)
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue
        
        # Update diff counter
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        
        new_collisions = _collision_count(list(diff_counter.elements()))
        
        # New scope
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < old_val:
            # Need to recompute
            new_scope = _scope(current)
        
        old_c = cost(current_scope, collisions)
        new_c = cost(new_scope, new_collisions)
        
        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob
        
        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best
>>>>>>> REPLACE

