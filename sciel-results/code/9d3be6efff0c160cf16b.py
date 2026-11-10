"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _sidon_sums(selected):
    """Compute all pair sums (including diagonal) for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    return sums

def _can_add(x, selected, sums):
    """Check if x can be added to selected without violating Sidon property."""
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal in [1,N]."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(x, selected, sums):
            return False
    return True

def _greedy_build(order, N):
    """Build a maximal Sidon set by greedily adding elements in given order."""
    selected = []
    sums = set()
    for x in order:
        if _can_add(x, selected, sums):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    # Saturate: add any remaining elements that can be added
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x not in set(selected) and _can_add(x, selected, sums):
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
                changed = True
    return selected

def _local_search(selected, N, deadline):
    """Local search: try swaps to reduce size while maintaining maximality."""
    selected = list(selected)
    sums = _sidon_sums(selected)
    best_size = len(selected)
    best_set = list(selected)
    
    # Precompute which elements can be added to the "reduced" set after removing one element
    rng = random.Random(123)
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        current_size = len(selected)
        
        # Try removing a random element and see if we can still make it maximal
        # with fewer elements
        idx = rng.randrange(current_size)
        removed = selected[idx]
        
        # Build reduced set
        reduced = selected[:idx] + selected[idx+1:]
        if not reduced:
            continue
        
        reduced_sums = _sidon_sums(reduced)
        
        # Check if reduced is still a valid Sidon set (it should be, as subset)
        # Now try to extend reduced to maximal, but prefer NOT adding back 'removed'
        # and try to get a smaller total size
        
        # First, check if reduced is already maximal
        if _is_maximal(reduced, reduced_sums):
            # Great! We found a smaller maximal set
            best_size = len(reduced)
            best_set = list(reduced)
            selected = list(reduced)
            sums = reduced_sums
            continue
        
        # Try to extend reduced greedily, avoiding 'removed' if possible
        ext_selected = list(reduced)
        ext_sums = reduced_sums
        ext_set = set(ext_selected)
        
        # Try to add elements in an order that avoids 'removed'
        candidates = [x for x in range(1, N + 1) if x not in ext_set]
        # Sort: prefer elements that are not 'removed', and randomize among them
        rng.shuffle(candidates)
        # Move 'removed' to the end if present
        if removed in candidates:
            candidates.remove(removed)
            candidates.append(removed)
        
        for x in candidates:
            if _can_add(x, ext_selected, ext_sums):
                ext_selected.append(x)
                for a in ext_selected[:-1]:
                    ext_sums.add(x + a)
                ext_sums.add(2 * x)
        
        # Check if the extended set is maximal
        if _is_maximal(ext_selected, ext_sums) and len(ext_selected) < best_size:
            best_size = len(ext_selected)
            best_set = list(ext_selected)
            selected = list(ext_selected)
            sums = ext_sums
        
        # Also try: remove element, then try to add a DIFFERENT element
        # that blocks more, potentially leading to a smaller maximal set
        # This is a 1-swap: remove one, add one (different)
        if current_size > 1:
            reduced2 = list(selected[:idx] + selected[idx+1:])
            reduced2_sums = _sidon_sums(reduced2)
            reduced2_set = set(reduced2)
            
            # Find elements that can be added to reduced2
            addable = []
            for x in range(1, N + 1):
                if x in reduced2_set:
                    continue
                if _can_add(x, reduced2, reduced2_sums):
                    addable.append(x)
            
            # Try a few random addable elements
            rng.shuffle(addable)
            for x in addable[:5]:
                new_selected = reduced2 + [x]
                new_sums = reduced2_sums
                for a in reduced2:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                
                # Check if this new set is maximal
                if _is_maximal(new_selected, new_sums) and len(new_selected) < best_size:
                    best_size = len(new_selected)
                    best_set = list(new_selected)
                    selected = list(new_selected)
                    sums = new_sums
                    break
    
    return best_set, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s kill
    
    best_size = N + 1
    best_set = None
    
    # Phase 1: Get initial maximal sets via greedy with different orderings
    rng = random.Random(42)
    
    # Ascending order
    asc_set = _greedy_build(list(range(1, N + 1)), N)
    if len(asc_set) < best_size and _is_maximal(asc_set, _sidon_sums(asc_set)):
        best_size = len(asc_set)
        best_set = list(asc_set)
    
    # Descending order
    desc_set = _greedy_build(list(range(N, 0, -1)), N)
    if len(desc_set) < best_size and _is_maximal(desc_set, _sidon_sums(desc_set)):
        best_size = len(desc_set)
        best_set = list(desc_set)
    
    # Several random orderings for initial diversity
    for i in range(50):
        if time.time() > deadline - 30.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_build(order, N)
        if len(candidate) < best_size and _is_maximal(candidate, _sidon_sums(candidate)):
            best_size = len(candidate)
            best_set = list(candidate)
    
    # Phase 2: Local search from the best set found
    if best_set is not None:
        improved_set, improved_size = _local_search(best_set, N, deadline)
        if improved_size < best_size:
            best_size = improved_size
            best_set = improved_set
    
    # Phase 3: More local search from different starting points if time remains
    if time.time() < deadline - 5.0:
        for i in range(10):
            if time.time() > deadline - 2.0:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            candidate = _greedy_build(order, N)
            if len(candidate) <= best_size:
                improved_set, improved_size = _local_search(candidate, N, deadline)
                if improved_size < best_size:
                    best_size = improved_size
                    best_set = improved_set
    
    if best_set is None:
        best_set = _greedy_build(list(range(1, N + 1)), N)
    
    return best_set
# EVOLVE-BLOCK-END
