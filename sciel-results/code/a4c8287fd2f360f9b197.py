"""Optimized solver for 'No element dividing a sum of three others'."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

MAX_N = 192
MAX_K = 48
TIME_LIMIT = 85.0  # safety margin before 90s kill

def _can_add(x, selected, selected_set):
    """Check if x can be added to selected without violating constraints.
    We only need to check:
    1. x does not divide sum of any 3 distinct other elements in (selected + [x])
    2. No existing element a in selected divides sum of any 3 distinct others in (selected + [x])
       that include x (since sums not including x were already valid)
    """
    # Check 1: x doesn't divide sum of any 3 from selected
    # (since x is new, the "other" elements for x are all in selected)
    if len(selected) >= 3:
        for combo in combinations(selected, 3):
            s = combo[0] + combo[1] + combo[2]
            if s % x == 0:
                return False
    
    # Check 2: for each a in selected, a doesn't divide sum of 3 distinct others that include x
    # The 3 others must be distinct from a, and include x, so we pick 2 from selected \ {a}
    for a in selected:
        if a == x:
            continue
        # Pick 2 elements from selected that are not a (x is implicit)
        # Need exactly 2 from selected \ {a}
        others = [v for v in selected if v != a]
        if len(others) >= 2:
            for pair in combinations(others, 2):
                s = x + pair[0] + pair[1]
                if s % a == 0:
                    return False
    
    return True

def _greedy_order(order, selected, selected_set):
    """Add elements in given order if possible."""
    for x in order:
        if x in selected_set:
            continue
        if _can_add(x, selected, selected_set):
            selected.append(x)
            selected_set.add(x)

def _try_remove_and_replace(selected, selected_set, candidate):
    """Try to remove one element and add candidate to improve or maintain size."""
    for i, a in enumerate(selected):
        new_selected = [v for j, v in enumerate(selected) if j != i]
        new_set = selected_set - {a}
        if _can_add(candidate, new_selected, new_set):
            return new_selected, new_set
    return None

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction - try multiple orderings
    best_selected = []
    best_set = set()
    
    # Strategy 1: Descending order (original approach, but faster check)
    selected = []
    selected_set = set()
    for x in range(MAX_N, 0, -1):
        if _can_add(x, selected, selected_set):
            selected.append(x)
            selected_set.add(x)
    if len(selected) > len(best_selected):
        best_selected = selected[:]
        best_set = selected_set.copy()
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return best_selected
    
    # Strategy 2: Randomized greedy with multiple restarts
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Create a random order with some bias
        # Mix of large and medium numbers
        pool = list(range(50, MAX_N + 1))  # Focus on larger numbers
        random.shuffle(pool)
        
        selected = []
        selected_set = set()
        
        # First, greedily add from shuffled pool
        for x in pool:
            if _can_add(x, selected, selected_set):
                selected.append(x)
                selected_set.add(x)
        
        # Then try to fill in smaller numbers
        for x in range(49, 0, -1):
            if x not in selected_set and _can_add(x, selected, selected_set):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_selected):
            best_selected = selected[:]
            best_set = selected_set.copy()
        
        # Early stop if we hit max
        if len(best_selected) >= MAX_K:
            break
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 3:
        return best_selected
    
    # Phase 2: Local improvement - try swapping elements
    # Try to improve by removing one element and adding a different one
    improved = True
    while improved and time.time() - start_time < TIME_LIMIT - 3:
        improved = False
        current = best_selected[:]
        current_set = best_set.copy()
        
        # Try all candidates not in current
        for candidate in range(1, MAX_N + 1):
            if candidate in current_set:
                continue
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            result = _try_remove_and_replace(current, current_set, candidate)
            if result is not None:
                new_selected, new_set = result
                # Only accept if size doesn't decrease (it won't, since we remove 1 add 1)
                # But we might want to prefer larger elements for flexibility
                if len(new_selected) >= len(current):
                    current = new_selected
                    current_set = new_set
                    improved = True
        
        if len(current) > len(best_selected):
            best_selected = current
            best_set = current_set
    
    # Phase 3: More aggressive randomized search
    for trial in range(500):
        if time.time() - start_time > TIME_LIMIT - 2:
            break
        
        # Start from best and perturb
        selected = best_selected[:]
        selected_set = best_set.copy()
        
        # Remove 1-3 random elements
        n_remove = random.randint(1, 3)
        for _ in range(n_remove):
            if selected:
                idx = random.randint(0, len(selected) - 1)
                removed = selected.pop(idx)
                selected_set.discard(removed)
        
        # Try to add back random candidates
        candidates = [x for x in range(1, MAX_N + 1) if x not in selected_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if _can_add(x, selected, selected_set):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_selected):
            best_selected = selected[:]
            best_set = selected_set.copy()
            if len(best_selected) >= MAX_K:
                break
    
    return best_selected
# EVOLVE-BLOCK-END
