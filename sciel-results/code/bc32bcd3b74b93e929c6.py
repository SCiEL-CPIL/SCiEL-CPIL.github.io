"""
Optimized small maximal Sidon set in [1,256].

Strategy:
1. Multiple greedy heuristics with different scoring functions
2. Aggressive local search: removal + swap operations
3. Efficient blocking analysis using sum-set membership
4. Diversified random restarts within time budget
"""
import random
import time

N = 256

def _compute_sums(sel):
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    return sums

def _is_sidon(sel):
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(sel, sums):
    """Check if sel is inclusion-maximal in [1,N]."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # x can be added if: 2x not in sums AND x+a not in sums for all a in sel
        if 2 * x in sums:
            continue
        blocked = False
        for a in sel:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_with_score(order, score_fn, sel, sums):
    """
    Greedily add elements from order, choosing the one with best score
    among all currently-addable candidates.
    """
    for x in order:
        # Check if x can be added
        if 2 * x in sums:
            continue
        can_add = True
        for a in sel:
            if (x + a) in sums:
                can_add = False
                break
        if can_add:
            sel.append(x)
            sums.add(2 * x)
            for a in sel:
                sums.add(x + a)
    return sel, sums

def _greedy_smart(candidates, sel, sums, rng, strategy="balance"):
    """
    Greedy that at each step picks the best candidate from a pool.
    strategy: 'block' (maximize blocking), 'unblocked' (minimize own blocking), 'balance'
    """
    while candidates:
        # Find all addable candidates
        addable = []
        for x in candidates:
            if 2 * x in sums:
                continue
            ok = True
            for a in sel:
                if (x + a) in sums:
                    ok = False
                    break
            if ok:
                addable.append(x)
        
        if not addable:
            break
        
        if len(addable) == 1:
            x = addable[0]
        else:
            # Score each addable candidate
            scores = []
            for x in addable:
                # How many other candidates would x block?
                block_count = 0
                for y in addable:
                    if y == x:
                        continue
                    # Would y be blocked after adding x?
                    # y is blocked if 2y in new_sums or y+a in new_sums for some a in new_sel
                    # new_sums = sums ∪ {2x} ∪ {x+a for a in sel∪{x}}
                    # y is blocked if (2y in sums or 2y == 2x) or (y+a in sums for some a in sel) or (y+x in sums or y+x == 2x)
                    # Simplify: check if y would conflict
                    if 2 * y == 2 * x:  # y == x, skip
                        continue
                    # Check if y becomes blocked
                    if 2 * y in sums:
                        block_count += 1
                        continue
                    y_blocked = False
                    for a in sel:
                        if (y + a) in sums:
                            y_blocked = True
                            break
                    if not y_blocked:
                        # Check y+x
                        if (y + x) in sums:
                            y_blocked = True
                        elif 2 * x == y + x:  # y == x
                            y_blocked = True
                        else:
                            # y+x is new, but check if it conflicts with existing sums
                            # Actually we just need: is y+x in sums? Already checked.
                            # Also 2y vs x+a for a in sel: if 2y == x+a for some a, that's a conflict
                            # But that would mean y is already blocked via 2y in sums... no, 2y not in sums.
                            # The new sums include x+a for a in sel. If 2y == x+a, then after adding x, 
                            # 2y is in sums, so y is blocked.
                            for a in sel:
                                if 2 * y == x + a:
                                    y_blocked = True
                                    break
                    if y_blocked:
                        block_count += 1
                
                # How "free" is x? (fewer existing conflicts = more flexible)
                # Count how many sums x+a hits (should be 0 since x is addable)
                # Instead, count how many future candidates x could conflict with
                conflict_with_future = 0
                for y in addable:
                    if y == x:
                        continue
                    if (x + y) in sums:
                        conflict_with_future += 1
                    if 2 * y in sums:
                        conflict_with_future += 1
                
                if strategy == "block":
                    score = block_count
                elif strategy == "unblocked":
                    score = -conflict_with_future
                else:  # balance
                    score = block_count - 0.5 * conflict_with_future
                scores.append((score, x))
            
            # Pick best (highest score)
            scores.sort(reverse=True)
            x = scores[0][1]
        
        # Add x
        sel.append(x)
        sums.add(2 * x)
        for a in sel:
            sums.add(x + a)
        
        # Remove x from candidates
        candidates.remove(x)
    
    return sel, sums

def _local_search_removal(sel, sums, deadline):
    """Try to reduce size by removing elements while maintaining maximality."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(sel)):
            if time.time() > deadline:
                break
            x_to_remove = sel[i]
            new_sel = sel[:i] + sel[i+1:]
            new_sums = _compute_sums(new_sel)
            if _is_maximal(new_sel, new_sums):
                sel = new_sel
                sums = new_sums
                improved = True
                break  # Restart with smaller set
    return sel, sums

def _local_search_swap(sel, sums, deadline):
    """
    Try to reduce size by swapping one element for another.
    If we can replace x with y such that the new set is still maximal and Sidon,
    and the new set is smaller... well swap keeps same size.
    Instead: try removing x and adding y where y was previously blocked, 
    but the net effect allows further removals.
    
    Better: try all pairs (remove x, add y) where y not in sel, and check if
    resulting set is maximal. If yes, we have same size but different structure,
    which may allow further removals.
    """
    improved = True
    sel_set = set(sel)
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(sel)):
            if time.time() > deadline:
                break
            x = sel[i]
            # Try removing x
            sel_without_x = sel[:i] + sel[i+1:]
            sums_without_x = _compute_sums(sel_without_x)
            
            # Now try adding each missing element y
            for y in range(1, N + 1):
                if y in sel_set and y != x:
                    continue
                if y == x:
                    continue
                # Check if y can be added to sel_without_x
                if 2 * y in sums_without_x:
                    continue
                ok = True
                for a in sel_without_x:
                    if (y + a) in sums_without_x:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Try the swap: sel_without_x + [y]
                new_sel = sel_without_x + [y]
                new_sel_set = set(new_sel)
                new_sums = _compute_sums(new_sel)
                
                # Check if this new set is maximal
                if _is_maximal(new_sel, new_sums):
                    # Same size, but different structure. Keep it if it enables further reduction.
                    # Actually, just accept it and let removal search try again.
                    sel = new_sel
                    sums = new_sums
                    sel_set = new_sel_set
                    improved = True
                    break
            if improved:
                break
    return sel, sums

def _combined_local_search(sel, sums, deadline):
    """Alternate removal and swap local search."""
    while time.time() < deadline:
        before_size = len(sel)
        sel, sums = _local_search_removal(sel, sums, deadline)
        if len(sel) < before_size:
            before_size = len(sel)
        sel, sums = _local_search_swap(sel, sums, deadline)
        sel, sums = _local_search_removal(sel, sums, deadline)
        if len(sel) == before_size:
            break
    return sel, sums

def _greedy_basic(order, sel, sums):
    """Simple greedy: add elements in given order if possible."""
    for x in order:
        if 2 * x in sums:
            continue
        can_add = True
        for a in sel:
            if (x + a) in sums:
                can_add = False
                break
        if can_add:
            sel.append(x)
            sums.add(2 * x)
            for a in sel:
                sums.add(x + a)
    return sel, sums

def construct():
    start_time = time.time()
    deadline = start_time + 28  # 2s safety margin
    
    best_set = None
    best_size = float('inf')
    
    def try_update(sel, sums):
        nonlocal best_set, best_size
        if sel and len(sel) < best_size:
            # Verify
            if _is_sidon(sel) and _is_maximal(sel, sums):
                best_size = len(sel)
                best_set = list(sel)
                return True
        return False
    
    # Strategy 1: Ascending greedy
    if time.time() < deadline:
        sel, sums = _greedy_basic(list(range(1, N + 1)), [], set())
        try_update(sel, sums)
    
    # Strategy 2: Descending greedy
    if time.time() < deadline:
        sel, sums = _greedy_basic(list(range(N, 0, -1)), [], set())
        try_update(sel, sums)
    
    # Strategy 3: Random greedy with smart selection
    rng = random.Random(12345)
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random order for smart greedy
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        strategy = rng.choice(["block", "unblocked", "balance"])
        sel, sums = _greedy_smart(candidates, [], set(), rng, strategy)
        
        if sel:
            try_update(sel, sums)
            
            # Local search
            if time.time() < deadline - 2:
                sel, sums = _combined_local_search(sel, sums, deadline - 1)
                try_update(sel, sums)
    
    # Strategy 4: More random restarts with basic greedy + local search
    for trial in range(500):
        if time.time() > deadline - 3:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        sel, sums = _greedy_basic(order, [], set())
        
        if sel:
            try_update(sel, sums)
            
            if time.time() < deadline - 1:
                sel, sums = _combined_local_search(sel, sums, deadline - 0.5)
                try_update(sel, sums)
    
    if best_set is None:
        return [1]
    
    return best_set
