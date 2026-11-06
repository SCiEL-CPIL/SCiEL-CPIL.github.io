"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_3ap_with(selected, x):
    """Check if adding x to selected creates a 3-AP. O(|selected|)."""
    for a in selected:
        # x is middle: 2x = a + b => b = 2x - a
        b = 2*x - a
        if b != a and b in selected:
            return True
        # a is middle: 2a = b + x => b = 2a - x
        b = 2*a - x
        if b != a and b in selected:
            return True
    return False

def _reciprocal_sum(selected):
    """Compute sum of reciprocals as float."""
    return sum(1.0/x for x in selected)

def _get_conflicts(selected, x):
    """Return set of elements in selected that form a 3-AP with x."""
    conflicts = set()
    for a in selected:
        b = 2*x - a
        if b != a and b in selected:
            conflicts.add(a)
            conflicts.add(b)
        b = 2*a - x
        if b != a and b in selected:
            conflicts.add(a)
            conflicts.add(b)
    return conflicts

def _greedy_by_reciprocal():
    """Greedy: add elements in order of largest reciprocal (smallest x)."""
    selected = set()
    for x in range(1, 257):
        if not _has_3ap_with(selected, x):
            selected.add(x)
    return selected

def _greedy_randomized(seed_val):
    """Greedy with randomization for diversity."""
    import random
    rng = random.Random(seed_val)
    selected = set()
    candidates = list(range(1, 257))
    weights = [1.0/x for x in candidates]
    total_w = sum(weights)
    while candidates and time.time() < _deadline:
        r = rng.random() * total_w
        cum = 0
        idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                idx = i
                break
        x = candidates[idx]
        if not _has_3ap_with(selected, x):
            selected.add(x)
        candidates.pop(idx)
        weights.pop(idx)
        total_w -= weights[idx] if idx < len(weights) else 0
    return selected

def _local_search(initial, deadline):
    """Efficient local search with 1-add, 1-swap, and 2-swap moves using precomputed conflicts."""
    selected = set(initial)
    best_sum = _reciprocal_sum(selected)
    best_set = set(selected)
    
    # Phase 1: Add all possible elements (fast pass)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 257):
            if x not in selected and not _has_3ap_with(selected, x):
                selected.add(x)
                changed = True
    best_sum = _reciprocal_sum(selected)
    best_set = set(selected)
    
    # Precompute conflict sets for all missing elements
    missing = [x for x in range(1, 257) if x not in selected]
    conflict_cache = {}
    for x in missing:
        conflict_cache[x] = _get_conflicts(selected, x)
    
    # Phase 2: 1-swap moves (remove y, add x) - use conflict cache
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Recompute conflicts for missing elements after each improvement
        missing = [x for x in range(1, 257) if x not in selected]
        conflict_cache = {}
        for x in missing:
            conflict_cache[x] = _get_conflicts(selected, x)
        
        for x in missing:
            if time.time() > deadline:
                return list(best_set)
            conflicts = conflict_cache[x]
            if not conflicts:
                continue
            for y in conflicts:
                if time.time() > deadline:
                    return list(best_set)
                new_set = selected - {y}
                # Check if x can be added to new_set
                if not _has_3ap_with(new_set, x):
                    new_set.add(x)
                    new_sum = _reciprocal_sum(new_set)
                    if new_sum > best_sum + 1e-15:
                        best_sum = new_sum
                        best_set = new_set
                        selected = set(new_set)
                        improved = True
                        break  # Restart loop after improvement
    
    # Phase 3: 2-swap moves (remove y1, y2, add x1, x2)
    # Focus on small missing numbers with high reciprocal value
    missing = [x for x in range(1, 257) if x not in selected]
    missing.sort(key=lambda x: 1.0/x, reverse=True)
    top_missing = missing[:30]
    
    # Precompute conflicts for top missing
    conflict_cache = {}
    for x in top_missing:
        conflict_cache[x] = _get_conflicts(selected, x)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [x for x in range(1, 257) if x not in selected]
        missing.sort(key=lambda x: 1.0/x, reverse=True)
        top_missing = missing[:30]
        
        conflict_cache = {}
        for x in top_missing:
            conflict_cache[x] = _get_conflicts(selected, x)
        
        for i, x1 in enumerate(top_missing):
            if time.time() > deadline:
                return list(best_set)
            if x1 in selected:
                continue
            conflicts1 = conflict_cache[x1]
            if not conflicts1:
                continue
            
            for x2 in top_missing[i+1:]:
                if time.time() > deadline:
                    return list(best_set)
                if x2 in selected:
                    continue
                conflicts2 = conflict_cache[x2]
                if not conflicts2:
                    continue
                
                # Check if x1 and x2 would form a 3-AP with each other
                b_check = 2*x1 - x2
                if b_check in selected or b_check == x1 or b_check == x2:
                    continue
                b_check = 2*x2 - x1
                if b_check in selected or b_check == x1 or b_check == x2:
                    continue
                b_check = 2*x1 - x2
                if 1 <= b_check <= 256 and b_check != x1 and b_check != x2:
                    # They form a 3-AP with b_check, but that's ok as long as b_check not in set
                    pass
                
                all_conflicts = conflicts1 | conflicts2
                conflict_list = sorted(all_conflicts, key=lambda y: -1.0/y)  # Prefer removing large y
                
                for ci in range(min(len(conflict_list), 15)):
                    if time.time() > deadline:
                        return list(best_set)
                    y1 = conflict_list[ci]
                    for cj in range(ci+1, min(len(conflict_list), 15)):
                        if time.time() > deadline:
                            return list(best_set)
                        y2 = conflict_list[cj]
                        new_set = selected - {y1, y2}
                        if not _has_3ap_with(new_set, x1) and not _has_3ap_with(new_set, x2):
                            new_set.add(x1)
                            new_set.add(x2)
                            new_sum = _reciprocal_sum(new_set)
                            if new_sum > best_sum + 1e-15:
                                best_sum = new_sum
                                best_set = new_set
                                selected = set(new_set)
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    
    # Phase 4: Final polish - try all single additions and swaps one more time
    if time.time() < deadline:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, 257):
                if x not in selected and not _has_3ap_with(selected, x):
                    selected.add(x)
                    changed = True
        
        # Final 1-swap pass
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in selected:
                continue
            conflicts = _get_conflicts(selected, x)
            for y in conflicts:
                if time.time() > deadline:
                    break
                new_set = selected - {y}
                if not _has_3ap_with(new_set, x):
                    new_set.add(x)
                    new_sum = _reciprocal_sum(new_set)
                    if new_sum > best_sum + 1e-15:
                        best_sum = new_sum
                        best_set = new_set
                        selected = set(new_set)
    
    return list(best_set)

def construct():
    global _deadline
    _deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Try multiple initializations
    best_result = None
    best_score = 0.0
    
    # Greedy by reciprocal (ascending)
    init1 = _greedy_by_reciprocal()
    result1 = _local_search(init1, _deadline)
    score1 = _reciprocal_sum(result1)
    if score1 > best_score:
        best_score = score1
        best_result = result1
    
    # A few randomized greedy starts with more time allocation
    remaining_time = _deadline - time.time()
    time_per_start = remaining_time / 3.0
    
    for seed_val in [42, 123, 7]:
        if time.time() > _deadline:
            break
        init_r = _greedy_randomized(seed_val)
        start_time = time.time()
        local_deadline = min(_deadline, start_time + time_per_start)
        result_r = _local_search(init_r, local_deadline)
        score_r = _reciprocal_sum(result_r)
        if score_r > best_score:
            best_score = score_r
            best_result = result_r
    
    return best_result if best_result else init1
# EVOLVE-BLOCK-END
