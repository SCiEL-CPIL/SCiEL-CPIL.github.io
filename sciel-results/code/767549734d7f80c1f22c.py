"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add(selected, x):
    """
    Check if x can be added to selected without violating constraints.
    1. x must not divide sum of any 3 distinct elements in selected.
    2. Each a in selected must not divide sum of any 3 distinct elements in selected U {x} 
       that are distinct from a. The new sums involving x are x + s1 + s2 for distinct s1, s2 in selected.
    """
    if not selected:
        return True
    
    # Check 1: x divides sum of 3 distinct in selected
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            return False
            
    # Check 2: For each a in selected, a divides (x + s1 + s2) for distinct s1, s2 in selected
    # Note: s1, s2 must be distinct from a (which they are, as they are in selected and we pick 2 from selected)
    # and distinct from x (x is new)
    # Also s1 != s2
    for a in selected:
        # We need to check if there exist distinct s1, s2 in selected such that (x + s1 + s2) % a == 0
        # Optimization: iterate pairs
        for s1, s2 in combinations(selected, 2):
            if (x + s1 + s2) % a == 0:
                return False
    return True

def _is_valid_set(selected):
    """
    Verify the entire set is valid. Used for debugging or final check.
    """
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_build(order):
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    """
    Try to improve the set by swapping elements or adding/removing.
    Since we want max cardinality, we primarily try to add elements.
    If we can't add, we might try to swap a smaller element for a larger one to "free up" divisibility constraints?
    Actually, larger numbers are generally "safer".
    Let's try to replace an element with a larger one not in the set, if it maintains validity and doesn't reduce size.
    Or just try to add more elements from the remaining pool.
    """
    selected_set = set(selected)
    available = [x for x in range(1, 193) if x not in selected_set]
    # Sort available by descending value, as larger values are less likely to cause conflicts
    available.sort(reverse=True)
    
    improved = True
    while improved:
        improved = False
        # Try to add any available element
        for x in available:
            if _can_add(selected, x):
                selected.append(x)
                selected_set.add(x)
                available.remove(x)
                improved = True
                break
        
        if time.time() > time_limit:
            break
            
        # If no element could be added, try swapping
        # This is expensive, so we limit it
        if not improved:
            # Try to replace an element in selected with a larger one from available
            # This might allow adding more later? Or just maintain size with better properties?
            # Actually, if we can't add, we are stuck at this size unless we swap.
            # Swapping might not increase size immediately, but might allow future additions.
            # Given time constraints, we might skip complex swaps or do a limited random swap.
            pass
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    best_set = []
    
    # 1. Deterministic Greedy (Descending)
    order = list(range(192, 0, -1))
    s1 = _greedy_build(order)
    if len(s1) > len(best_set):
        best_set = s1
        
    if time.time() > deadline:
        return best_set

    # 2. Randomized Greedy with Local Search
    # We try multiple random orders, biased towards larger numbers
    for i in range(100): # Limit restarts
        if time.time() > deadline:
            break
            
        # Create a random order, but maybe shuffle only the smaller numbers?
        # Or just shuffle all.
        order = list(range(1, 193))
        random.shuffle(order)
        
        s = _greedy_build(order)
        
        # Local search: try to add more
        s = _local_search(s, deadline)
        
        if len(s) > len(best_set):
            best_set = s
            
    # 3. Try to improve best_set with local search again
    if time.time() < deadline:
        best_set = _local_search(best_set, deadline)
        
    return best_set
# EVOLVE-BLOCK-END
