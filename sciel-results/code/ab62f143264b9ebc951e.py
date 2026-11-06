# EVOLVE-BLOCK-START
def _is_valid_sidon(s):
    """Check if sorted list s is a valid Sidon set (all pairwise sums distinct)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            v = s[i] + s[j]
            if v in sums:
                return False
            sums.add(v)
    return True

def _get_sums(s):
    """Get all pairwise sums (i <= j) of sorted list s."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sums.add(s[i] + s[j])
    return sums

def _can_add(s, sums, x):
    """Check if x can be added to Sidon set s with existing sums."""
    for y in s:
        if x + y in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _add_to_sums(sums, s, x):
    """Add x to sums set, assuming compatibility."""
    for y in s:
        sums.add(x + y)
    sums.add(2 * x)

def _greedy_random_valid(start_set, rng):
    """
    Greedy construction that at each step picks a random valid candidate
    from the set of all currently valid candidates.
    """
    s = list(start_set)
    sums = _get_sums(s)
    s_set = set(s)
    
    while True:
        valid_candidates = []
        for x in range(1, 285):
            if x in s_set:
                continue
            if _can_add(s, sums, x):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # Pick a random valid candidate
        x = rng.choice(valid_candidates)
        s.append(x)
        s_set.add(x)
        _add_to_sums(sums, s, x)
        
    s.sort()
    return s

def _local_search(s, deadline, rng):
    """Local search with plateau exploration."""
    s = sorted(s)
    sums = _get_sums(s)
    best = list(s)
    best_size = len(s)
    
    # Phase 1: Try to add any missing element
    changed = True
    while changed:
        changed = False
        for x in range(1, 285):
            if x in s:
                continue
            if _can_add(s, sums, x):
                s.append(x)
                _add_to_sums(sums, s, x)
                s.sort()
                best = list(s)
                best_size = len(s)
                changed = True
                if time.time() > deadline:
                    return best
    
    # Phase 2: Plateau search and kicks
    for _ in range(3000): # Reduced iterations to allow more restarts
        if time.time() > deadline:
            return best
        
        # Check if we can add
        added = False
        for x in range(1, 285):
            if x in s:
                continue
            if _can_add(s, sums, x):
                s.append(x)
                _add_to_sums(sums, s, x)
                s.sort()
                best = list(s)
                best_size = len(s)
                added = True
                break
        
        if added:
            continue
        
        # If no add, try neutral move or kick
        move_type = rng.random()
        
        if move_type < 0.5:
            # Neutral move: remove 1, add 1
            if len(s) < 2:
                continue
            i = rng.randrange(len(s))
            removed = s[i]
            s_temp = s[:i] + s[i+1:]
            sums_temp = _get_sums(s_temp)
            
            # Find a valid replacement
            replacements = []
            for x in range(1, 285):
                if x in s_temp or x == removed:
                    continue
                if _can_add(s_temp, sums_temp, x):
                    replacements.append(x)
            
            if replacements:
                x_new = rng.choice(replacements)
                # Ensure we actually change the set
                if x_new != removed:
                    s_new = s_temp + [x_new]
                    s_new.sort()
                    s = s_new
                    sums = _get_sums(s)
                    
        else:
            # Kick: remove 1, add 2
            if len(s) < 2:
                continue
            i = rng.randrange(len(s))
            removed = s[i]
            s_temp = s[:i] + s[i+1:]
            sums_temp = _get_sums(s_temp)
            
            added_count = 0
            # Try to add 2 elements
            for x in range(1, 285):
                if x in s_temp or x == removed:
                    continue
                if _can_add(s_temp, sums_temp, x):
                    s_temp.append(x)
                    _add_to_sums(sums_temp, s_temp, x)
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count == 2:
                s = s_temp
                sums = sums_temp
                s.sort()
                if len(s) > best_size:
                    best = list(s)
                    best_size = len(s)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    start_set = [1, 2, 5, 11]
    best = list(start_set)
    best_size = len(start_set)
    
    rng = random.Random(42)
    
    # Use the new random valid greedy
    num_restarts = 200 # Increase restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        rng_restarts = random.Random(r * 12345 + 67890)
        
        s = _greedy_random_valid(start_set, rng_restarts)
        
        # Local search
        s = _local_search(s, deadline, rng_restarts)
        
        if len(s) > best_size:
            best = list(s)
            best_size = len(s)
            
            # Early exit if we hit 20
            if best_size >= 20:
                break
    
    # Final local search on the best found
    if time.time() < deadline - 2:
        best = _local_search(best, deadline, rng)
        
    if not _is_valid_sidon(best):
        best = [1, 2, 5, 11]
        sums = _get_sums(best)
        for x in range(12, 285):
            if _can_add(best, sums, x):
                best.append(x)
                _add_to_sums(sums, best, x)
        best.sort()
        
    return best
# EVOLVE-BLOCK-END
