"""Vertex-cover-inspired greedy blocking search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check all unordered pair sums a+b (a<=b) are distinct."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in s:
                return False
            s.add(t)
    return True

def _compute_sums(selected):
    """Compute all pair sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums, sel_set):
    """Check if x can be added to selected while preserving Sidon property."""
    if x in sel_set:
        return False
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_fast(selected, sums, sel_set):
    """Check if selected is inclusion-maximal (no element can be added)."""
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums, sel_set):
            return False
    return True

def _greedy_vertex_cover(seed_elements, rng, time_limit, use_degree=True):
    """
    Vertex-cover-inspired greedy: maintain an explicit addable set,
    score each candidate by how many remaining candidates it would block
    (degree in the conflict graph), pick the highest-degree node.
    Returns (final_set, final_sums, final_sel_set) or None.
    """
    selected = list(seed_elements)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    
    # Initialize addable set
    addable = set()
    for x in range(1, N + 1):
        if x not in sel_set and _is_addable(x, selected, sums, sel_set):
            addable.add(x)
    
    while addable:
        if time.time() > time_limit:
            return None
        
        candidates = list(addable)
        
        if use_degree:
            # Vertex-cover heuristic: pick candidate with highest "degree"
            # (most remaining candidates it would block)
            best_x = None
            best_score = -1
            best_newly_blocked = set()
            
            # Shuffle for tie-breaking diversity
            rng.shuffle(candidates)
            
            for x in candidates:
                # Compute new sums from adding x
                new_sums = set()
                for a in selected:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                
                # Count how many remaining addable elements become blocked
                newly_blocked = set()
                for y in addable:
                    if y == x:
                        continue
                    # Check if y would be blocked after adding x
                    blocked = False
                    for a in selected:
                        if y + a in sums:
                            blocked = True
                            break
                    if not blocked:
                        if 2 * y in sums:
                            blocked = True
                    if not blocked:
                        # Check against new sums from x
                        if y + x in sums:
                            blocked = True
                        elif 2 * y in new_sums:
                            blocked = True
                        else:
                            for a in selected:
                                if y + a in new_sums:
                                    blocked = True
                                    break
                    if blocked:
                        newly_blocked.add(y)
                
                score = len(newly_blocked)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_newly_blocked = newly_blocked
            
            if best_x is None:
                break
        else:
            # Random fallback
            best_x = rng.choice(candidates)
            # Compute newly blocked
            new_sums = set()
            for a in selected:
                new_sums.add(best_x + a)
            new_sums.add(2 * best_x)
            best_newly_blocked = set()
            for y in addable:
                if y == best_x:
                    continue
                blocked = False
                for a in selected:
                    if y + a in sums:
                        blocked = True
                        break
                if not blocked:
                    if 2 * y in sums:
                        blocked = True
                if not blocked:
                    if y + best_x in sums:
                        blocked = True
                    elif 2 * y in new_sums:
                        blocked = True
                    else:
                        for a in selected:
                            if y + a in new_sums:
                                blocked = True
                                break
                if blocked:
                    best_newly_blocked.add(y)
        
        # Add best_x
        for a in selected:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        selected.append(best_x)
        sel_set.add(best_x)
        addable.discard(best_x)
        
        # Remove newly blocked elements
        addable -= best_newly_blocked
        
        # Recheck remaining addable elements (some may have become invalid)
        still_addable = set()
        for x in addable:
            if _is_addable(x, selected, sums, sel_set):
                still_addable.add(x)
        addable = still_addable
    
    return selected, sums, sel_set

def _local_search_aggressive(selected, sums, sel_set, rng, time_limit):
    """
    Aggressive local search: try removing elements, 1-swaps, and 2-swaps
    to find smaller maximal Sidon sets.
    Returns (shrunk_set, shrunk_sums, shrunk_sel_set) or None.
    """
    selected = list(selected)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    best_size = len(selected)
    
    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_limit:
            return None
        
        improved = False
        
        # Phase 1: Try removing single elements
        for i in range(len(selected)):
            if time.time() > time_limit:
                return None
            
            x = selected[i]
            new_selected = selected[:i] + selected[i+1:]
            new_sel_set = sel_set - {x}
            new_sums = _compute_sums(new_selected)
            
            if _is_maximal_fast(new_selected, new_sums, new_sel_set):
                selected = new_selected
                sums = new_sums
                sel_set = new_sel_set
                best_size = len(selected)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        if len(selected) >= 2:
            for i in range(len(selected)):
                if time.time() > time_limit:
                    return None
                
                removed = selected[i]
                base_selected = selected[:i] + selected[i+1:]
                base_sel_set = sel_set - {removed}
                base_sums = _compute_sums(base_selected)
                
                # Find addable elements to the base
                candidates = []
                for x in range(1, N + 1):
                    if x not in base_sel_set and _is_addable(x, base_selected, base_sums, base_sel_set):
                        candidates.append(x)
                
                if not candidates:
                    continue
                
                # Try candidates, prefer those that make the set maximal
                rng.shuffle(candidates)
                found = False
                for x in candidates[:15]:
                    test_selected = base_selected + [x]
                    test_sel_set = base_sel_set | {x}
                    test_sums = set(base_sums)
                    for a in base_selected:
                        test_sums.add(x + a)
                    test_sums.add(2 * x)
                    
                    if len(test_selected) < best_size and _is_maximal_fast(test_selected, test_sums, test_sel_set):
                        selected = test_selected
                        sums = test_sums
                        sel_set = test_sel_set
                        best_size = len(selected)
                        improved = True
                        found = True
                        break
                
                if found:
                    break
        
        if improved:
            continue
        
        # Phase 3: Try 2-swaps (remove two, add two) - more expensive
        if len(selected) >= 3:
            n = len(selected)
            # Sample a few pairs to try
            pair_attempts = min(20, n * (n-1) // 2)
            pairs_tried = set()
            for _ in range(pair_attempts):
                if time.time() > time_limit:
                    return None
                
                i, j = rng.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key in pairs_tried:
                    continue
                pairs_tried.add(key)
                
                removed1 = selected[i]
                removed2 = selected[j]
                base_selected = [x for idx, x in enumerate(selected) if idx != i and idx != j]
                base_sel_set = sel_set - {removed1, removed2}
                base_sums = _compute_sums(base_selected)
                
                # Find addable elements
                candidates = []
                for x in range(1, N + 1):
                    if x not in base_sel_set and _is_addable(x, base_selected, base_sums, base_sel_set):
                        candidates.append(x)
                
                if len(candidates) < 2:
                    continue
                
                # Try pairs of candidates
                rng.shuffle(candidates)
                limit = min(10, len(candidates))
                found = False
                for a in range(limit):
                    for b in range(a + 1, limit):
                        x, y = candidates[a], candidates[b]
                        test_selected = base_selected + [x, y]
                        test_sel_set = base_sel_set | {x, y}
                        test_sums = set(base_sums)
                        for a2 in base_selected:
                            test_sums.add(x + a2)
                            test_sums.add(y + a2)
                        test_sums.add(2 * x)
                        test_sums.add(2 * y)
                        test_sums.add(x + y)
                        
                        if len(test_selected) < best_size and _is_maximal_fast(test_selected, test_sums, test_sel_set):
                            selected = test_selected
                            sums = test_sums
                            sel_set = test_sel_set
                            best_size = len(selected)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if found:
                    break
        
        if not improved:
            break
    
    return selected, sums, sel_set

def _verify_maximal(selected):
    """Verify selected is Sidon and inclusion-maximal."""
    if not _is_sidon(selected):
        return False
    sums = _compute_sums(selected)
    sel_set = set(selected)
    return _is_maximal_fast(selected, sums, sel_set)

def _force_maximal(selected):
    """Force a set to be maximal by adding all addable elements."""
    selected = list(selected)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x not in sel_set and _is_addable(x, selected, sums, sel_set):
                for a in selected:
                    sums.add(x + a)
                sums.add(2 * x)
                selected.append(x)
                sel_set.add(x)
                changed = True
    
    return selected, sums, sel_set

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin before 30s limit
    
    # Get initial upper bound from simple greedy (ascending)
    simple_selected, simple_sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)
    
    best_set = simple_selected
    best_sums = _compute_sums(best_set)
    best_sel_set = set(best_set)
    best_size = len(best_set)
    
    # Verify it's maximal; if not, force it
    if not _is_maximal_fast(best_set, best_sums, best_sel_set):
        best_set, best_sums, best_sel_set = _force_maximal(best_set)
        best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Main search loop with vertex-cover-inspired greedy
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        
        # Vary seed strategy based on attempt number
        if attempts % 10 == 0:
            # Every 10th: start empty with random strategy
            seed_size = 0
            seed_elements = []
            use_degree = True
        elif attempts % 10 == 1:
            # Small random seed
            seed_size = rng.randint(1, 3)
            seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
            use_degree = True
        elif attempts % 10 == 2:
            # Larger seed
            seed_size = rng.randint(3, 6)
            seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
            use_degree = True
        elif attempts % 10 == 3:
            # Random strategy (no degree)
            seed_size = rng.randint(0, 2)
            seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
            use_degree = False
        elif attempts % 10 == 4:
            # Bias toward small elements
            seed_size = rng.randint(0, 3)
            seed_elements = rng.sample(range(1, N // 2 + 1), min(seed_size, N // 2))
            use_degree = True
        elif attempts % 10 == 5:
            # Bias toward large elements
            seed_size = rng.randint(0, 3)
            seed_elements = rng.sample(range(N // 2 + 1, N + 1), min(seed_size, N // 2))
            use_degree = True
        else:
            # Standard random
            seed_size = rng.randint(0, 4)
            seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
            use_degree = True
        
        # Check seed is Sidon
        seed_ok = True
        seed_sums = set()
        for i, a in enumerate(seed_elements):
            for b in seed_elements[i:]:
                s = a + b
                if s in seed_sums:
                    seed_ok = False
                    break
                seed_sums.add(s)
            if not seed_ok:
                break
        
        if not seed_ok:
            continue
        
        # Greedy construction
        result = _greedy_vertex_cover(seed_elements, rng, deadline - 1.0, use_degree)
        if result is None:
            continue
        
        final_set, final_sums, final_sel_set = result
        
        # Verify maximality
        if not _is_maximal_fast(final_set, final_sums, final_sel_set):
            # Force maximal
            final_set, final_sums, final_sel_set = _force_maximal(final_set)
        
        if len(final_set) < best_size:
            # Try local search to shrink further
            if time.time() < deadline - 1.5:
                ls_result = _local_search_aggressive(final_set, final_sums, final_sel_set, rng, deadline - 0.5)
                if ls_result is not None:
                    sel, sm, ss = ls_result
                    if _is_maximal_fast(sel, sm, ss) and len(sel) < best_size:
                        best_size = len(sel)
                        best_set = sel
                        best_sums = sm
                        best_sel_set = ss
                else:
                    best_size = len(final_set)
                    best_set = final_set
                    best_sums = final_sums
                    best_sel_set = final_sel_set
            else:
                best_size = len(final_set)
                best_set = final_set
                best_sums = final_sums
                best_sel_set = final_sel_set
        elif len(final_set) == best_size:
            # Same size, try local search anyway
            if time.time() < deadline - 1.5:
                ls_result = _local_search_aggressive(final_set, final_sums, final_sel_set, rng, deadline - 0.5)
                if ls_result is not None:
                    sel, sm, ss = ls_result
                    if _is_maximal_fast(sel, sm, ss) and len(sel) < best_size:
                        best_size = len(sel)
                        best_set = sel
                        best_sums = sm
                        best_sel_set = ss
    
    # Final local search on the best found
    if time.time() < deadline - 0.5 and len(best_set) > 5:
        ls_result = _local_search_aggressive(best_set, best_sums, best_sel_set, rng, deadline - 0.1)
        if ls_result is not None:
            sel, sm, ss = ls_result
            if _is_maximal_fast(sel, sm, ss) and len(sel) < best_size:
                best_set = sel
                best_size = len(sel)
    
    return best_set
# EVOLVE-BLOCK-END
