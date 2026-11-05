import numpy as np
import random

N = 256

def _can_add(x, in_set, sums):
    """Check if x can be added to the set defined by in_set and sums."""
    if in_set[x]:
        return False
    if sums[2 * x]:
        return False
    # Check conflicts with existing elements
    elems = np.where(in_set)[0]
    if len(elems) > 0:
        # Check if any (x + e) is in sums
        if np.any(sums[x + elems]):
            return False
    return True

def _update_sums_add(x, in_set, sums):
    """Update sums array after adding x."""
    elems = np.where(in_set)[0]
    sums[2 * x] = True
    if len(elems) > 0:
        sums[x + elems] = True

def _update_sums_remove(x, in_set, sums):
    """Update sums array after removing x.
    Note: This is tricky because sums might be shared.
    We must rebuild or carefully decrement.
    For simplicity and correctness, we rebuild sums after removal in the reduction phase.
    """
    pass

def _rebuild_sums(in_set):
    """Rebuild sums array from in_set."""
    sums = np.zeros(2 * N + 1, dtype=bool)
    elems = np.where(in_set)[0]
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            sums[elems[i] + elems[j]] = True
    return sums

def _is_maximal(in_set, sums):
    """Check if the set is maximal."""
    for x in range(1, N + 1):
        if not in_set[x]:
            if _can_add(x, in_set, sums):
                return False
    return True

def _greedy_init():
    """Generate a random maximal Sidon set."""
    in_set = np.zeros(N + 1, dtype=bool)
    sums = np.zeros(2 * N + 1, dtype=bool)
    order = list(range(1, N + 1))
    random.shuffle(order)
    
    for x in order:
        if _can_add(x, in_set, sums):
            in_set[x] = True
            _update_sums_add(x, in_set, sums)
            
    return in_set, sums

def _try_reduce(in_set, sums):
    """Try to remove elements while maintaining maximality."""
    improved = True
    while improved:
        improved = False
        elems = np.where(in_set)[0].tolist()
        random.shuffle(elems)
        
        for x in elems:
            # Try removing x
            in_set_test = in_set.copy()
            in_set_test[x] = False
            sums_test = _rebuild_sums(in_set_test)
            
            if _is_maximal(in_set_test, sums_test):
                in_set = in_set_test
                sums = sums_test
                improved = True
                break # Restart loop as set changed
                
    return in_set, sums

def construct():
    best_in_set = None
    best_size = N + 1
    
    for restart in range(200):
        in_set, sums = _greedy_init()
        in_set, sums = _try_reduce(in_set, sums)
        
        size = np.sum(in_set)
        if size < best_size:
            best_size = size
            best_in_set = in_set.copy()
            
    if best_in_set is None:
        return list(range(1, 10))
        
    return np.where(best_in_set)[0].tolist()
