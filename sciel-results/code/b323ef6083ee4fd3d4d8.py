"""Optimized backtracking/greedy blocking-priority search for small maximal Sidon sets in [1,256]."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START
def _compute_sums_np(selected):
    """Compute all pair sums using numpy for speed. Returns a set of sums."""
    arr = np.array(selected)
    i, j = np.triu_indices(len(arr))
    sums = arr[i] + arr[j]
    return set(sums.tolist())

def _is_addable_fast(x, selected, sums):
    """Check if x can be added. Uses set lookup."""
    if 2 * x in sums:
        return False
    for a in selected:
        if x + a in sums:
            return False
    return True

def _is_maximal_fast(selected, sums):
    """Check inclusion-maximality."""
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

def _greedy_construct(seed_elements, rng, time_limit):
    """Greedy: pick element that blocks the most addable candidates."""
    selected = list(seed_elements)
    sums = _compute_sums_np(selected)
    sel_set = set(selected)
    
    # Build initial addable set
    addable = set()
    for x in range(1, N + 1):
        if x not in sel_set:
            if 2 * x not in sums:
                blocked = False
                for a in selected:
                    if x + a in sums:
                        blocked = True
                        break
                if not blocked:
                    addable.add(x)
    
    while addable:
        if time.time() > time_limit:
            return None
        
        best_x = None
        best_score = -1
        best_newly_blocked = set()
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for x in candidates:
            # New sums from adding x
            new_sums = set()
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            
            # Count newly blocked
            newly_blocked = set()
            for y in addable:
                if y == x:
                    continue
                # y is blocked if:
                # - (y+x) in sums (old) or new_sums
                # - 2y in new_sums
                # - exists a in selected: (y+a) in new_sums
                if y + x in sums or y + x in new_sums:
                    newly_blocked.add(y)
                    continue
                if 2 * y in new_sums:
                    newly_blocked.add(y)
                    continue
                blocked = False
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
        
        sums.update([best_x + a for a in selected] + [2 * best_x])
        selected.append(best_x)
        sel_set.add(best_x)
        addable.discard(best_x)
        addable -= best_newly_blocked
    
    return selected, sums

def _local_search_remove(selected, sums, rng, time_limit):
    """Remove elements while maintaining maximality."""
    selected = list(selected)
    sums = _compute_sums_np(selected)
    sel_set = set(selected)
    
    improved = True
    while improved:
        if time.time() > time_limit:
            return None
        
        improved = False
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for i in indices:
            x = selected[i]
            
            # Sums removed when x is removed
            removed_sums = set()
            for j in range(len(selected)):
                if selected[j] >= x:
                    removed_sums.add(x + selected[j])
            removed_sums.add(2 * x)
            
            new_sums = sums - removed_sums
            new_sel_set = sel_set - {x}
            
            # Check maximality after removal
            is_maximal = True
            for y in range(1, N + 1):
                if y in new_sel_set:
                    continue
                if 2 * y in new_sums:
                    continue
                blocked = False
                for a in selected:
                    if a == x:
                        continue
                    if y + a in new_sums:
                        blocked = True
                        break
                if not blocked:
                    is_maximal = False
                    break
            
            if is_maximal:
                selected.pop(i)
                sums = new_sums
                sel_set = new_sel_set
                improved = True
                break
    
    return selected, sums

def _verify_maximal(selected):
    """Verify Sidon and inclusion-maximal."""
    sums = _compute_sums_np(selected)
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in s:
                return False
            s.add(t)
    return _is_maximal_fast(selected, sums)

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin
    
    # Initial greedy ascending
    simple_selected = []
    simple_sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)
    
    best_set = simple_selected
    best_size = len(best_set)
    
    # Ensure maximality
    if not _verify_maximal(best_set):
        changed = True
        while changed:
            changed = False
            sums = _compute_sums_np(best_set)
            sel_set = set(best_set)
            for x in range(1, N + 1):
                if x not in sel_set:
                    if _is_addable_fast(x, best_set, sums):
                        best_set.append(x)
                        changed = True
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Try diverse strategies
    while time.time() < deadline - 2.0:
        # Random seed
        seed_size = rng.randint(0, 6)
        seed_elements = rng.sample(range(1, N + 1), min(seed_size, N))
        
        # Verify seed is Sidon
        seed_ok = True
        seed_sums = set()
        for i, a in enumerate(seed_elements):
            for b in seed_elements[i:]:
                s_val = a + b
                if s_val in seed_sums:
                    seed_ok = False
                    break
                seed_sums.add(s_val)
            if not seed_ok:
                break
        if not seed_ok:
            continue
        
        result = _greedy_construct(seed_elements, rng, deadline - 1.5)
        if result is None:
            continue
        
        final_set, final_sums = result
        
        if not _verify_maximal(final_set):
            continue
        
        current_size = len(final_set)
        
        # Local search to shrink
        if current_size < best_size or (current_size <= best_size and rng.random() < 0.5):
            shrunk_result = _local_search_remove(final_set, final_sums, rng, deadline - 1.0)
            if shrunk_result is not None:
                shrunk_set, shrunk_sums = shrunk_result
                if _verify_maximal(shrunk_set):
                    if len(shrunk_set) < best_size:
                        best_size = len(shrunk_set)
                        best_set = shrunk_set
                    current_size = len(shrunk_set)
        
        if current_size < best_size:
            best_size = current_size
            best_set = final_set
    
    return best_set
# EVOLVE-BLOCK-END
