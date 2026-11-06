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
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal."""
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
    if 2 * x in sums:
        return False, selected, sums
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
    """
    Greedily add elements until maximal.
    Strategy: Choose the element that blocks the most other unblocked elements.
    """
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # Choose element to add: prefer one that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        # Limit evaluation to a random subset of unblocked if list is very long to save time?
        # N=256, unblocked can be up to 256. 256*256 checks is ~65k ops, very fast.
        for x in unblocked:
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
            
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in selected:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if not y_blocked and y + x in new_sums:
                    y_blocked = True
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            best_x = rng.choice(unblocked)
        
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            break
    
    return selected, sums

def _local_search_reduce(selected, sums, rng, deadline):
    """
    Try to remove elements while maintaining maximality.
    Also try swaps if removal is blocked.
    """
    selected = list(selected)
    sums = set(sums)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
            # Check Sidon validity
            new_sums = set()
            valid = True
            n = len(candidate)
            for a_idx in range(n):
                a = candidate[a_idx]
                for b_idx in range(a_idx, n):
                    s = a + candidate[b_idx]
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

def _swap_local_search(selected, sums, rng, deadline):
    """
    If we can't remove an element, try swapping it with a different element
    that keeps the set maximal and potentially allows future removals.
    This is a heuristic to escape local minima in size.
    """
    selected = list(selected)
    sums = set(sums)
    sel_set = set(selected)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            
            elem_to_remove = selected[i]
            candidate_base = selected[:i] + selected[i+1:]
            
            # Find elements that can be added to candidate_base
            # and result in a maximal set
            # We want to find a replacement 'y' such that
            # 1. candidate_base + [y] is a valid Sidon set
            # 2. candidate_base + [y] is maximal
            # 3. y is not in candidate_base
            
            # To save time, only try a few random candidates for replacement
            # Or try all unblocked elements in candidate_base?
            # Let's try a random sample of potential replacements from [1, N]
            
            # First, check if candidate_base itself is maximal (handled by reduce step)
            # If not, we need to add something.
            
            # Compute sums of candidate_base
            base_sums = set()
            valid_base = True
            n_base = len(candidate_base)
            for a_idx in range(n_base):
                a = candidate_base[a_idx]
                for b_idx in range(a_idx, n_base):
                    s = a + candidate_base[b_idx]
                    if s in base_sums:
                        valid_base = False
                        break
                    base_sums.add(s)
                if not valid_base:
                    break
            
            if not valid_base:
                continue
            
            # Try to find a y to add
            # We want y to block as many unblocked elements as possible
            # But primarily, we just need the resulting set to be maximal.
            
            # Heuristic: Try a few random y's
            for _ in range(10):
                if time.time() > deadline:
                    break
                y = rng.randint(1, N)
                if y in candidate_base:
                    continue
                
                # Check if y can be added to candidate_base
                ok, new_sel, new_sums = _try_add(candidate_base, base_sums, y)
                if not ok:
                    continue
                
                # Check if new_sel is maximal
                if _is_maximal(new_sel, new_sums):
                    # Accept swap
                    selected = new_sel
                    sums = new_sums
                    improved = True
                    break
            
            if improved:
                break
    
    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Random initial Sidon set + greedy maximize + local search
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        # Start with a small random seed set
        target_size = rng.randint(6, 12)
        for x in candidates:
            if len(selected) >= target_size:
                break
            ok, selected, sums = _try_add(selected, sums, x)
        
        if not selected:
            continue
        
        # Greedy maximize to get a maximal set
        selected, sums = _greedy_maximize(selected, sums, rng, deadline)
        
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sums):
            continue
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
        
        # Local search to reduce size
        if time.time() < deadline:
            reduced_sel, reduced_sums = _local_search_reduce(selected, sums, rng, deadline)
            if _is_valid_sidon(reduced_sel) and _is_maximal(reduced_sel, reduced_sums):
                if len(reduced_sel) < best_size:
                    best_size = len(reduced_sel)
                    best = list(reduced_sel)
            
            # Try swap local search if we are stuck
            if time.time() < deadline and len(reduced_sel) > 12:
                swapped_sel, swapped_sums = _swap_local_search(reduced_sel, reduced_sums, rng, deadline)
                if _is_valid_sidon(swapped_sel) and _is_maximal(swapped_sel, swapped_sums):
                    if len(swapped_sel) < best_size:
                        best_size = len(swapped_sel)
                        best = list(swapped_sel)
                    
                    # Try reducing again after swap
                    if time.time() < deadline:
                        final_sel, final_sums = _local_search_reduce(swapped_sel, swapped_sums, rng, deadline)
                        if _is_valid_sidon(final_sel) and _is_maximal(final_sel, final_sums):
                            if len(final_sel) < best_size:
                                best_size = len(final_sel)
                                best = list(final_sel)
        
        if best_size <= 12:
            break
    
    # Fallback: If no good result found, return a greedy maximal set
    if best is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
