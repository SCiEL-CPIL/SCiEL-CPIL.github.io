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
            s = a + b
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

def _greedy_set_cover(selected, sums, rng, deadline):
    """
    Structural change: Model as Greedy Set Cover.
    Universe: Elements in [1, N] not in selected.
    Sets: For each candidate x, the set of elements y in Universe that would be blocked if x is added.
    Goal: Cover all elements in Universe with minimum number of sets (elements added).
    """
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        unblocked_set = set(unblocked)
        
        # Find candidate x that covers the most currently unblocked elements
        best_x = None
        max_covered = 0
        best_covered_set = set()
        
        # Optimization: Only consider x that is valid to add
        # We can iterate through all x in 1..N not in selected
        # But we only care about those that block *something*
        
        sel_set = set(selected)
        
        # To speed up, we can pre-calculate which x are valid additions
        # But checking validity is O(|selected|). 
        # N=256, |selected| ~ 20. So 256*20 = 5120 ops per check.
        # We do this for each x in 1..256. 256 * 5120 = 1.3M ops.
        # This is fine for one step.
        
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            
            # Check if x is valid to add
            if 2 * x in sums:
                continue
            valid = True
            for a in selected:
                if x + a in sums:
                    valid = False
                    break
            if not valid:
                continue
            
            # Calculate coverage: which unblocked elements does x block?
            covered = []
            for y in unblocked:
                # y is blocked by x if:
                # 1. 2y is in new_sums (i.e., 2y == 2x or 2y == x+a for some a)
                # 2. y+a is in new_sums (i.e., y+a == 2x or y+a == x+b for some b)
                
                # Check 2y
                if 2 * y == 2 * x:
                    covered.append(y)
                    continue
                if any(2 * y == x + a for a in selected):
                    covered.append(y)
                    continue
                
                # Check y+a
                y_blocked = False
                # y+a == 2x
                if y + selected[0] if selected else 0: # dummy check to avoid empty list error if selected empty, but selected is never empty here
                    pass
                # Actually, simpler:
                # y is blocked if (y+2x in sums) OR (y+x+a in sums for some a) ... wait.
                # The condition for y being blocked by adding x is:
                # Adding x creates sums: {2x} U {x+a | a in S}
                # y is blocked if 2y is in (Sums U {2x} U {x+a}) OR (y+a) is in (Sums U {2x} U {x+a}) for some a in S U {x}.
                
                # Let's just simulate the block check for y against the hypothetical new set S' = S U {x}
                # y is blocked in S' if:
                # 1. 2y in Sums(S')
                # 2. y+a in Sums(S') for some a in S'
                
                # Sums(S') = Sums(S) U {2x} U {x+a | a in S}
                
                # Check 1: 2y in Sums(S')
                # 2y in Sums(S) is false because y is currently unblocked (if 2y was in Sums(S), y would be blocked).
                # So check if 2y == 2x OR 2y == x+a for some a in S.
                if 2 * y == 2 * x:
                    covered.append(y)
                    continue
                found_2y = False
                for a in selected:
                    if 2 * y == x + a:
                        found_2y = True
                        break
                if found_2y:
                    covered.append(y)
                    continue
                
                # Check 2: y+a in Sums(S') for some a in S' = S U {x}
                # Case 2a: a=x. y+x in Sums(S')?
                # y+x in Sums(S) is false (y unblocked implies y+a not in Sums(S) for a in S, but x is not in S).
                # Wait, if y is unblocked in S, it means:
                # 2y not in Sums(S) AND (y+a) not in Sums(S) for all a in S.
                
                # So we need to check if y+x is in Sums(S') or y+a is in Sums(S') for a in S.
                
                # y+x in Sums(S')?
                # Sums(S') contains Sums(S). y+x not in Sums(S) because x not in S.
                # Sums(S') contains 2x. Is y+x == 2x? => y==x. Impossible as x in S, y not in S.
                # Sums(S') contains x+a. Is y+x == x+a? => y==a. Impossible.
                # So y+x is NOT in Sums(S').
                
                # y+a in Sums(S') for a in S?
                # We know y+a not in Sums(S).
                # So we need to check if y+a == 2x OR y+a == x+b for some b in S.
                
                blocked_by_x = False
                # Check y+a == 2x
                for a in selected:
                    if y + a == 2 * x:
                        blocked_by_x = True
                        break
                if not blocked_by_x:
                    # Check y+a == x+b
                    for a in selected:
                        for b in selected:
                            if y + a == x + b:
                                blocked_by_x = True
                                break
                        if blocked_by_x:
                            break
                            
                if blocked_by_x:
                    covered.append(y)
            
            if len(covered) > max_covered:
                max_covered = len(covered)
                best_x = x
                best_covered_set = set(covered)
        
        if best_x is None:
            # No valid x covers anything? Should not happen if unblocked is not empty and set is Sidon.
            # If unblocked is not empty, there must be some x that can be added?
            # Actually, if the set is maximal, unblocked is empty.
            # If not maximal, unblocked is not empty.
            # Is it possible that no x in 1..N can be added?
            # If S is Sidon, and unblocked is not empty, then there exists y in unblocked.
            # y can be added. So best_x should at least be y (covering y itself? No, adding y blocks y? No, y becomes selected).
            # Wait, if we add y, y is no longer unblocked. So y covers itself?
            # My coverage definition: "elements y in Universe that would be blocked if x is added".
            # If x=y, then y is added to S. It is no longer in Universe.
            # Does adding y "block" y? No, it removes it from the set of missing elements.
            # The goal is to make the set maximal, i.e., Universe becomes empty.
            # So covering y means y is no longer in Universe.
            # So if x=y, it covers y.
            
            # Let's re-verify the "covered" logic.
            # If x=y, then y is added.
            # My loop `for y in unblocked`:
            # If x=y:
            # 1. 2y == 2x -> True. So y is covered.
            # So best_x will at least be some y in unblocked.
            
            # If best_x is still None, it means no x is valid?
            # That implies S is not a Sidon set? Or we made a mistake.
            # Fallback: pick random unblocked
            best_x = rng.choice(unblocked)
        
        # Add best_x
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            break

    return selected, sums

def _local_search_reduce(selected, sums, rng, deadline):
    """
    Attempt to reduce size by swapping or removing.
    """
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
            # Check Sidon validity
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
            
            # Check Maximality
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
            else:
                # If not maximal, try to find a replacement
                unblocked = _get_unblocked(candidate, new_sums)
                if not unblocked:
                    continue
                
                # Try to add one element from unblocked to restore maximality
                # We want to add an element that is "efficient" (blocks many)
                # But we are trying to keep size same or reduce?
                # If we remove 1 and add 1, size is same.
                # If we remove 1 and add 0 (impossible if not maximal), size reduces.
                # So here we are doing a swap.
                
                # To minimize size, we prefer swaps that allow further reductions?
                # For now, just try to find any valid swap that doesn't increase size.
                # Since we removed one, adding one keeps size same.
                # We hope that the new set is "better" for future reductions.
                
                # Heuristic: Pick the unblocked element that blocks the most other unblocked elements
                best_swap = None
                max_block = -1
                
                for x in unblocked:
                    ok, new_sel, new_sum = _try_add(candidate, new_sums, x)
                    if not ok:
                        continue
                    if _is_maximal(new_sel, new_sum):
                        # Count how many unblocked (in candidate) are blocked by x
                        # This is complex to compute fast.
                        # Just accept the first valid maximal swap?
                        # Or random?
                        # Let's use random to explore
                        best_swap = x
                        break # Just take the first one to save time
                        
                if best_swap is not None:
                    ok, selected, sums = _try_add(candidate, new_sums, best_swap)
                    if ok:
                        improved = True
                        break

    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Strategy: Greedy Set Cover from small random starts
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Generate a random small Sidon seed
        # Smaller seeds might lead to more additions, but maybe better coverage structure?
        # Or larger seeds might be closer to minimal?
        # Minimal maximal Sidon sets are often small.
        # Let's try seeds of size 5-15.
        
        target_size = rng.randint(5, 15)
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if len(selected) >= target_size:
                break
            ok, selected, sums = _try_add(selected, sums, x)
            if ok:
                pass
        
        if not selected:
            continue
        
        # Apply Greedy Set Cover
        selected, sums = _greedy_set_cover(selected, sums, rng, deadline)
        
        # Verify
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sums):
            continue
        
        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best = list(selected)
            
            # If we found a small one, try to optimize it further
            if best_size <= 20:
                # Local search
                selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
                if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                    if len(selected2) < best_size:
                        best_size = len(selected2)
                        best = list(selected2)
        
        # Early termination
        if best_size <= 12:
            break

    # Strategy 2: Structured starts
    if time.time() < deadline:
        for base in range(1, 10):
            if time.time() > deadline:
                break
            selected = []
            sums = set()
            # Add elements with large gaps to ensure Sidon
            for x in range(base, N + 1, base * 2 + 1):
                ok, selected, sums = _try_add(selected, sums, x)
                if ok:
                    continue
            
            if selected:
                selected, sums = _greedy_set_cover(selected, sums, rng, deadline)
                if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best = list(selected)
                        
                        selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
                        if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                            if len(selected2) < best_size:
                                best_size = len(selected2)
                                best = list(selected2)

    if best is None:
        # Fallback
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
