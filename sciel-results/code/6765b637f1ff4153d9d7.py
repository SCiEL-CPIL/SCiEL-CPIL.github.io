"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from math import lcm

# EVOLVE-BLOCK-START

def _lcm_1_22():
    result = 1
    for i in range(1, 23):
        result = lcm(result, i)
    return result

_LCM = _lcm_1_22()

def _weights():
    return [_LCM // i for i in range(1, 23)]

def _is_dissociated(selected):
    """Check if selected indices (0-based, referring to [1..22]) form a dissociated set.
    
    A set is dissociated if all subset sums of the reciprocal values are distinct.
    Equivalently, all subset sums of the scaled weights (LCM // i) are distinct.
    """
    if not selected:
        return True
    
    # Get the scaled weights for the selected elements
    # selected contains 1-based values from 1 to 22
    weights = [_LCM // v for v in selected]
    n = len(weights)
    
    # Generate all 2^n subset sums and check for duplicates
    # For n up to ~12, 2^12 = 4096 which is fast
    # For larger n, we need a smarter approach
    
    if n <= 20:
        # Direct enumeration
        seen = set()
        seen.add(0)
        for mask in range(1, 1 << n):
            s = 0
            # Efficient subset sum via bit operations
            # Use iterative building
            pass
        
        # Better: use the fact that subset sums can be built incrementally
        sums = {0}
        for w in weights:
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    else:
        # For larger sets, use a more efficient check
        # Actually, let's just do the full check with early termination
        sums = set()
        sums.add(0)
        for w in weights:
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

def _try_extend(current_set, remaining, target_size, deadline):
    """Try to extend current_set with elements from remaining to reach target_size.
    
    Returns a valid dissociated set of size target_size, or None.
    Uses backtracking with pruning.
    """
    if time.time() > deadline:
        return None
    
    if len(current_set) == target_size:
        return list(current_set)
    
    needed = target_size - len(current_set)
    if needed > len(remaining):
        return None
    
    for i, v in enumerate(remaining):
        if time.time() > deadline:
            return None
        
        new_set = current_set + [v]
        # Check if new_set is still dissociated
        if _is_dissociated(new_set):
            result = _try_extend(new_set, remaining[i+1:], target_size, deadline)
            if result is not None:
                return result
    
    return None

def _greedy_construct(order, target_size, deadline):
    """Greedy construction: add elements in the given order if they maintain dissociation."""
    current = []
    for v in order:
        if time.time() > deadline:
            break
        if len(current) >= target_size:
            break
        new_set = current + [v]
        if _is_dissociated(new_set):
            current = new_set
    return current if current else None

def _random_search(deadline, best_size):
    """Random search for larger dissociated sets."""
    import random
    all_vals = list(range(1, 23))
    best = None
    best_sz = best_size
    
    while time.time() < deadline:
        # Try random subsets of increasing size
        for size in range(best_sz + 1, 23):
            if time.time() > deadline:
                break
            # Try a few random subsets of this size
            for _ in range(20):
                if time.time() > deadline:
                    break
                candidate = random.sample(all_vals, size)
                if _is_dissociated(candidate):
                    if best is None or len(candidate) > len(best):
                        best = candidate
                        best_sz = len(candidate)
                    break  # Found one of this size, try larger
    
    return best

def _systematic_search(deadline, best_size):
    """Systematic search: try to find sets of size best_size+1, best_size+2, etc."""
    all_vals = list(range(1, 23))
    best = None
    best_sz = best_size
    
    for target in range(best_sz + 1, 23):
        if time.time() > deadline:
            break
        
        # Try various orderings
        import random
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random ordering
            order = all_vals[:]
            random.shuffle(order)
            
            result = _greedy_construct(order, target, deadline)
            if result is not None and len(result) == target:
                best = result
                best_sz = target
                break
        
        if best_sz >= target:
            continue
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    # Start with a known good set
    initial = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    if not _is_dissociated(initial):
        initial = [1, 2, 3, 5, 7, 11, 13, 17]
    if not _is_dissociated(initial):
        initial = [1, 2, 3, 5, 7, 11, 13]
    if not _is_dissociated(initial):
        initial = [1, 2, 3, 5, 7, 11]
    if not _is_dissociated(initial):
        initial = [1, 2, 3, 5, 7]
    if not _is_dissociated(initial):
        initial = [1, 2, 3, 5]
    if not _is_dissociated(initial):
        initial = [1, 2, 3]
    if not _is_dissociated(initial):
        initial = [1, 2]
    if not _is_dissociated(initial):
        initial = [1]
    
    best = initial
    best_sz = len(best)
    
    # Phase 1: Greedy with multiple orderings
    import random
    all_vals = list(range(1, 23))
    
    # Try several deterministic orderings
    orderings = [
        sorted(all_vals),  # ascending
        sorted(all_vals, reverse=True),  # descending
        sorted(all_vals, key=lambda x: 1/x),  # by reciprocal (same as ascending)
        sorted(all_vals, key=lambda x: _LCM // x),  # by weight (ascending)
        sorted(all_vals, key=lambda x: _LCM // x, reverse=True),  # by weight descending
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy_construct(order, 22, deadline)
        if result is not None and len(result) > best_sz:
            best = result
            best_sz = len(result)
    
    # Phase 2: Random orderings
    for _ in range(100):
        if time.time() > deadline:
            break
        order = all_vals[:]
        random.shuffle(order)
        result = _greedy_construct(order, 22, deadline)
        if result is not None and len(result) > best_sz:
            best = result
            best_sz = len(result)
    
    # Phase 3: Random subset search for larger sizes
    result = _random_search(deadline, best_sz)
    if result is not None and len(result) > best_sz:
        best = result
        best_sz = len(result)
    
    # Phase 4: Systematic extension
    result = _systematic_search(deadline, best_sz)
    if result is not None and len(result) > best_sz:
        best = result
        best_sz = len(result)
    
    # Final validation
    if not _is_dissociated(best):
        # Fallback to a smaller known-good set
        for size in range(len(best)-1, 0, -1):
            for candidate in itertools.combinations(range(1, 23), size):
                if _is_dissociated(list(candidate)):
                    best = list(candidate)
                    break
            if _is_dissociated(best):
                break
        if not _is_dissociated(best):
            best = [1]
    
    return best

# EVOLVE-BLOCK-END
