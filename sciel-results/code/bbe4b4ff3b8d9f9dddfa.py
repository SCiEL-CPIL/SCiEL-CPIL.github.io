<<<<<<< SEARCH
def _can_add(x, selected, table):
    """Check if adding x to selected keeps the set valid."""
    for i in range(len(selected)):
        a = selected[i]
        for j in range(i + 1, len(selected)):
            b = selected[j]
            if not _is_valid_triple_fast(a, b, x, table):
                return False
    return True
=======
def _can_add(x, selected, table):
    """Check if adding x to selected keeps the set valid."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        ga_x = table[a][x]
        for j in range(i + 1, n):
            b = selected[j]
            if table[a][b] == ga_x == table[b][x]:
                return False
    return True
>>>>>>> REPLACE
<<<<<<< SEARCH
def _local_improve_swap(selected, candidates, table, deadline, max_swaps=200):
    """Try swap-based local search: remove one, add one or more."""
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        # Pick a random element to remove
        idx = random.randint(0, len(improved) - 1)
        removed = improved[idx]
        improved.pop(idx)
        
        # Try to add multiple excluded elements
        added_any = False
        remaining = [e for e in excluded if e not in improved]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if _can_add(x, improved, table):
                improved.append(x)
                added_any = True
        
        if not added_any or len(improved) < len(selected):
            # Revert
            improved.insert(idx, removed)
        else:
            # Update excluded list
            in_set = set(improved)
            excluded = [c for c in candidates if c not in in_set]
    
    return improved
=======
def _local_improve_swap(selected, candidates, table, deadline, max_swaps=200):
    """Try swap-based local search: remove one, add one or more."""
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        # Pick a random element to remove
        idx = random.randint(0, len(improved) - 1)
        removed = improved[idx]
        improved.pop(idx)
        
        # Try to add multiple excluded elements
        added_any = False
        remaining = [e for e in excluded if e not in improved]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if _can_add(x, improved, table):
                improved.append(x)
                added_any = True
        
        if not added_any or len(improved) <= len(selected):
            # Revert
            improved.insert(idx, removed)
        else:
            # Update excluded list
            in_set = set(improved)
            excluded = [c for c in candidates if c not in in_set]
    
    return improved
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Strategy 5: Many random greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 800
    for i in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order, table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, table, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved2 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=50)
                    if len(improved2) > best_len:
                        best_set, best_len = improved2, len(improved2)
=======
    # Strategy 5: Many random greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 800
    for i in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order, table)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, table, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved2 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=80)
                    if len(improved2) > best_len:
                        best_set, best_len = improved2, len(improved2)
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Final local improvement
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, table, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=100)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
=======
    # Final local improvement
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, table, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=150)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
        # Second round of local improvement if time permits
        if time.time() < deadline:
            improved3 = _local_improve_add(best_set, all_nums, table, deadline)
            if len(improved3) > best_len:
                best_set, best_len = improved3, len(improved3)
        if time.time() < deadline:
            improved4 = _local_improve_swap(best_set, all_nums, table, deadline, max_swaps=100)
            if len(improved4) > best_len:
                best_set, best_len = improved4, len(improved4)
>>>>>>> REPLACE
