import random
import time

N = 256

def _compute_sums(selected):
    """Compute all unordered pair sums (including a+a) for a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            sums.add(a + b)
    return sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + b if j > i else a + a
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal.
    For every x not in selected, adding x must create a conflict.
    x conflicts iff 2x in sums OR exists a in selected such that x+a in sums.
    """
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _try_add(selected, sums, x):
    """Try to add x to selected. Return (success, new_selected, new_sums)."""
    # Check 2x
    if 2 * x in sums:
        return False, selected, sums
    # Check x + a for all a in selected
    for a in selected:
        if x + a in sums:
            return False, selected, sums
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        new_sums.add(x + a)
    new_selected = selected + [x]
    return True, new_selected, new_sums

def _get_unblocked(selected, sums):
    """Return list of missing elements that can be added (not blocked)."""
    sel_set = set(selected)
    unblocked = []
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            unblocked.append(x)
    return unblocked

def _greedy_maximize(selected, sums, rng, deadline):
    """Greedily add elements until maximal, trying to minimize additions."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # Choose element to add: prefer one that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        for x in unblocked:
            # Simulate adding x
            new_sums = set(sums)
            valid = True
            if 2 * x in new_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for a in selected:
                    s = x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            if not valid:
                continue
            
            # Count how many other unblocked elements are now blocked
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                # Check if y is now blocked with new_sums
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in selected + [x]:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            # Fallback: pick random
            best_x = rng.choice(unblocked)
        
        # Add best_x
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            # Shouldn't happen, but break to avoid infinite loop
            break
    
    return selected, sums

def _local_search_reduce(selected, sums, rng, deadline):
    """Try to remove elements while maintaining maximality, or swap to reduce size."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            # Try removing selected[i]
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            # Recompute sums
            new_sums = set()
            valid = True
            for a_idx, a in enumerate(candidate):
                for b in candidate[a_idx:]:
                    s = a + b
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
    
    return selected, sums

def _hit_set_search(selected, sums, rng, deadline):
    """Model as minimum hitting set over conflict hyperedges.
    
    For each missing element x, define a hyperedge H_x = set of selected elements
    that would need to be removed (or new elements added) to make x insertable.
    We want to find a small set S such that for every x not in S, S "hits" H_x
    in the sense that x cannot be added to S.
    
    Greedy weighted set cover heuristic: iteratively pick the element that covers
    the most uncovered hyperedges.
    """
    selected = list(selected)
    sums = set(sums)
    sel_set = set(selected)
    
    # Compute initial blocking structure
    # For each missing x, determine which selected elements are "responsible" for blocking it
    # x is blocked if 2x in sums OR exists a in selected with x+a in sums
    
    # Build conflict hyperedges: for each missing x, H_x = {a in selected : x+a in sums} ∪ 
    #                                                              {a in selected : 2x = a+b for some a,b in selected}
    # Simplified: x is blocked by the sums that contain x+a or 2x
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing one element and see if we can maintain maximality
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
            # Recompute sums
            new_sums = set()
            valid = True
            for a_idx, a in enumerate(candidate):
                for b in candidate[a_idx:]:
                    s = a + b
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            
            # Check if candidate is still maximal
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
    
    return selected, sums

def _conflict_count(selected, sums, x):
    """Count how many selected elements are 'conflicting' with adding x."""
    sel_set = set(selected)
    if x in sel_set:
        return 0
    count = 0
    if 2 * x in sums:
        count += 1
    for a in selected:
        if x + a in sums:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Multiple strategies
    # Strategy 1: Random initial Sidon set, then greedy maximize
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate a random small Sidon set
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        # Try to build a small Sidon set of size ~15-25
        target_size = rng.randint(12, 25)
        for x in candidates:
            if len(selected) >= target_size:
                break
            ok, selected, sums = _try_add(selected, sums, x)
            if ok:
                pass
            else:
                continue
        
        if not selected:
            continue
        
        # Now greedily maximize
        selected, sums = _greedy_maximize(selected, sums, rng, deadline)
        
        # Verify
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sums):
            continue
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
        
        # Local search to try to reduce size
        if time.time() < deadline:
            selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
            if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                if len(selected2) < best_size:
                    best_size = len(selected2)
                    best = list(selected2)
            
            # Hit set search
            if time.time() < deadline:
                selected3, sums3 = _hit_set_search(selected2, sums2, rng, deadline)
                if _is_valid_sidon(selected3) and _is_maximal(selected3, sums3):
                    if len(selected3) < best_size:
                        best_size = len(selected3)
                        best = list(selected3)
        
        # Early termination if we find a very small set
        if best_size <= 12:
            break
    
    # Strategy 2: Start from known small structures
    if time.time() < deadline:
        # Try a structured approach: use elements spaced out
        for base in range(1, 10):
            if time.time() > deadline:
                break
            selected = []
            sums = set()
            # Add elements with large gaps
            for x in range(base, N + 1, base * 2 + 1):
                ok, selected, sums = _try_add(selected, sums, x)
                if ok:
                    continue
            if selected:
                selected, sums = _greedy_maximize(selected, sums, rng, deadline)
                if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best = list(selected)
                    if time.time() < deadline:
                        selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
                        if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                            if len(selected2) < best_size:
                                best_size = len(selected2)
                                best = list(selected2)
    
    if best is None:
        # Fallback: use the original greedy approach
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
