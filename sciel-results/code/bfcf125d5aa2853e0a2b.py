import random
import numpy as np

N = 256

def _get_sums(s):
    """Compute set of all pairwise sums a+b for a<=b in s."""
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.add(s[i] + s[j])
    return sums

def _is_maximal(s, n):
    """Check if Sidon set s is inclusion-maximal in [1, n]."""
    s_set = set(s)
    if not s_set:
        return False # Empty set is not maximal unless N=0? Task says nonempty.
    
    sums = _get_sums(s)
    
    for x in range(1, n + 1):
        if x in s_set:
            continue
        # Check if x can be added
        # New sums: {x+a for a in s} U {2x}
        # If any of these is in sums, x cannot be added.
        # If none are in sums, x CAN be added, so s is not maximal.
        
        # Optimization: Check 2x first
        if 2 * x in sums:
            continue
        for a in s:
            if x + a in sums:
                continue
        # If we reach here, no sum conflicted, so x can be added
        return False
    return True

def _greedy_maximal(order):
    """Build a maximal Sidon set by processing elements in given order."""
    selected = []
    sums = set()
    s_set = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        # Check internal conflicts in added
        if len(set(added)) != len(added):
            continue
        # Check against existing sums
        if sums.intersection(added):
            continue
        
        selected.append(x)
        sums.update(added)
        s_set.add(x)
        
    # The greedy process naturally produces a maximal set?
    # If we process ALL elements in order, and add if possible,
    # then at the end, no remaining element could be added (because we tried them all).
    # So yes, this is maximal.
    return selected

def construct():
    best_set = None
    best_size = N + 1
    
    # Random seed for reproducibility in evaluation? No, usually random is fine.
    # But let's use a fixed seed for the first few to ensure we don't fail completely?
    # The task says "random" is available.
    
    for restart in range(100):
        # 1. Generate a random maximal Sidon set
        order = list(range(1, N + 1))
        random.shuffle(order)
        current = _greedy_maximal(order)
        
        if not current:
            continue
            
        current_size = len(current)
        
        # 2. Local Search: Delete-While-Maximal
        # We try to remove elements one by one if the set remains maximal.
        # To avoid bias, we shuffle the removal order.
        
        improved = True
        while improved:
            improved = False
            # Shuffle a copy of the current set to try removals in random order
            to_try = list(current)
            random.shuffle(to_try)
            
            for x in to_try:
                if x not in current:
                    continue
                # Try removing x
                new_set = [y for y in current if y != x]
                # Check if new_set is maximal
                # Note: new_set is automatically Sidon since it's a subset.
                if _is_maximal(new_set, N):
                    current = new_set
                    current_size -= 1
                    improved = True
        
        if current_size < best_size:
            best_size = current_size
            best_set = list(current)
            
        # Optional: Try swaps to escape local minima?
        # Swapping doesn't change size, so it only helps if it allows subsequent deletions.
        # Given time constraints, simple deletion might be sufficient.
        
    if best_set is None:
        # Fallback
        order = list(range(1, N + 1))
        return _greedy_maximal(order)
        
    return best_set
