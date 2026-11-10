"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if not selected:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]. Assumes valid Sidon."""
    selected_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        can_add = True
        for a in selected:
            if x + a in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        if can_add:
            return False
    return True

def _greedy_build(order, N):
    """Greedy construction: add elements in given order if they don't conflict."""
    selected = []
    sums = set()
    for x in order:
        added_sums = set()
        can_add = True
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            selected.append(x)
            sums.update(added_sums)
    return selected

def _greedy_extend(base, N):
    """Greedily extend a valid Sidon set to maximal in [1,N]."""
    ext = list(base)
    ext_set = set(ext)
    ext_sums = set()
    for i, a in enumerate(ext):
        for j in range(i, len(ext)):
            ext_sums.add(a + ext[j])
    
    for x in range(1, N + 1):
        if x in ext_set:
            continue
        added_sums = set()
        can_add = True
        for a in ext:
            s = x + a
            if s in ext_sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in ext_sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            ext.append(x)
            ext_set.add(x)
            ext_sums.update(added_sums)
    return ext

def _random_small_sidon(size, N, rng):
    """
    Generate a random valid Sidon set of approximately 'size' elements.
    Uses rejection sampling with a small random subset of candidates to ensure sparsity.
    """
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # We only try to pick from a subset of candidates to force sparsity
    # This helps avoid the "dense" trap of standard greedy
    # Limit candidate pool to first 50% of shuffled list to encourage random spread
    pool = candidates[:N//2]
    
    for x in pool:
        if len(selected) >= size:
            break
        can_add = True
        added_sums = set()
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            selected.append(x)
            sums.update(added_sums)
            
    return selected

def _try_remove_and_extend(selected, N):
    """
    Try removing one element and greedily extending to maximal.
    Returns the smallest valid maximal set found, or None.
    """
    n = len(selected)
    if n <= 1:
        return None
    
    best = None
    best_len = n + 1
    
    # Sort removal candidates by heuristic: prefer removing elements that
    # are "central" (have many conflicts) to maximize room for re-extension
    indices = list(range(n))
    
    for i in indices:
        if time.time() > 58.0:
            break
        reduced = [selected[j] for j in range(n) if j != i]
        if not reduced:
            continue
        if not _is_valid_sidon(reduced):
            continue
        
        # Greedily extend reduced to maximal
        ext = _greedy_extend(reduced, N)
        
        if _is_valid_sidon(ext) and _is_maximal(ext, N):
            if len(ext) < best_len:
                best_len = len(ext)
                best = list(ext)
    
    return best

def _try_swap(selected, N):
    """
    Try swapping one element out and another in, then extend to maximal.
    Returns the smallest valid maximal set found, or None.
    """
    n = len(selected)
    if n <= 2:
        return None
    
    best = None
    best_len = n + 1
    selected_set = set(selected)
    
    for i in range(n):
        if time.time() > 58.0:
            break
        removed = selected[i]
        reduced = [selected[j] for j in range(n) if j != i]
        if not _is_valid_sidon(reduced):
            continue
        
        reduced_set = set(reduced)
        reduced_sums = set()
        for i2, a in enumerate(reduced):
            for j2 in range(i2, len(reduced)):
                reduced_sums.add(a + reduced[j2])
        
        # Find candidates that can replace 'removed'
        for x in range(1, N + 1):
            if time.time() > 58.0:
                break
            if x in reduced_set or x == removed:
                continue
            can_add = True
            for a in reduced:
                if x + a in reduced_sums:
                    can_add = False
                    break
            if can_add and 2 * x in reduced_sums:
                can_add = False
            if not can_add:
                continue
            
            # Build new set with x instead of removed
            new_base = reduced + [x]
            new_set = _greedy_extend(new_base, N)
            
            if _is_valid_sidon(new_set) and _is_maximal(new_set, N):
                if len(new_set) < best_len:
                    best_len = len(new_set)
                    best = list(new_set)
    
    return best

def _try_double_remove(selected, N):
    """
    Try removing two elements and greedily extending to maximal.
    Returns the smallest valid maximal set found, or None.
    """
    n = len(selected)
    if n <= 3:
        return None
    
    best = None
    best_len = n + 1
    
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > 58.0:
                break
            reduced = [selected[k] for k in range(n) if k != i and k != j]
            if len(reduced) < 2:
                continue
            if not _is_valid_sidon(reduced):
                continue
            
            ext = _greedy_extend(reduced, N)
            
            if _is_valid_sidon(ext) and _is_maximal(ext, N):
                if len(ext) < best_len:
                    best_len = len(ext)
                    best = list(ext)
    
    return best

def _try_shake(selected, N, rng, shake_size=3):
    """
    Randomly remove 'shake_size' elements and extend to maximal.
    This is a steepest-descent escape move.
    """
    n = len(selected)
    if n <= shake_size + 2:
        return None
        
    best = None
    best_len = n + 1
    
    # Try a few random shakes
    for _ in range(5):
        if time.time() > 58.0:
            break
        # Pick random indices to remove
        to_remove = rng.sample(range(n), min(shake_size, n-2))
        remove_set = set(to_remove)
        reduced = [selected[k] for k in range(n) if k not in remove_set]
        
        if len(reduced) < 2:
            continue
            
        # reduced is a subset of a valid Sidon set, so it is valid
        ext = _greedy_extend(reduced, N)
        
        if _is_valid_sidon(ext) and _is_maximal(ext, N):
            if len(ext) < best_len:
                best_len = len(ext)
                best = list(ext)
                
    return best

def construct():
    deadline = time.time() + 55.0  # safety margin
    rng = random.Random(42)
    
    best_set = None
    best_size = N + 1
    
    def _update_best(candidate):
        nonlocal best_set, best_size
        if candidate is not None and _is_valid_sidon(candidate) and _is_maximal(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = list(candidate)

    # Phase 1: Sparse Random Starts (Structural Change)
    # Generate small random Sidon sets and extend them.
    # This avoids the "dense" local minima found by standard greedy.
    for attempt in range(1000):
        if time.time() > deadline - 30.0:
            break
        
        # Try varying sizes of initial seeds
        seed_size = rng.randint(3, 12)
        seed = _random_small_sidon(seed_size, N, rng)
        
        if len(seed) < 2:
            continue
            
        # Extend to maximal
        candidate = _greedy_extend(seed, N)
        _update_best(candidate)
        
        if best_size <= 12:
            break
            
    # Fallback to standard greedy if sparse starts didn't find anything good
    if best_set is None or best_size > 20:
        baseline = _greedy_build(list(range(1, N + 1)), N)
        _update_best(baseline)
        
        desc = _greedy_build(list(range(N, 0, -1)), N)
        _update_best(desc)
        
        # Standard random greedy
        for attempt in range(200):
            if time.time() > deadline - 30.0:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            candidate = _greedy_build(order, N)
            _update_best(candidate)

    if best_set is None:
        # Absolute fallback
        best_set = _greedy_build(list(range(1, N + 1)), N)
        best_size = len(best_set)

    # Phase 2: Local Improvement Loop
    current = list(best_set)
    
    # We run local search on the current best.
    # We also periodically try to "shake" the current solution.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 1. Single removal
        result = _try_remove_and_extend(current, N)
        if result is not None and len(result) < len(current):
            current = result
            improved = True
            _update_best(current)
        
        # 2. Swap
        result = _try_swap(current, N)
        if result is not None and len(result) < len(current):
            current = result
            improved = True
            _update_best(current)
            
        # 3. Double removal
        result = _try_double_remove(current, N)
        if result is not None and len(result) < len(current):
            current = result
            improved = True
            _update_best(current)
            
        # 4. Shake (Random removal of 3 elements)
        # Only try shake if we are stuck or periodically
        if not improved and time.time() < deadline - 5.0:
             result = _try_shake(current, N, rng, shake_size=3)
             if result is not None and len(result) < len(current):
                 current = result
                 improved = True
                 _update_best(current)
                 
    # Phase 3: Final push with new sparse seeds if time remains
    if time.time() < deadline - 2.0:
        for attempt in range(100):
            if time.time() > deadline - 1.0:
                break
            seed_size = rng.randint(3, 10)
            seed = _random_small_sidon(seed_size, N, rng)
            if len(seed) < 2: continue
            candidate = _greedy_extend(seed, N)
            _update_best(candidate)
            if best_size <= 12: break

    return best_set
# EVOLVE-BLOCK-END
