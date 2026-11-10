"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_new_element(new_elem, existing_list):
    """Check if adding new_elem to existing_list is valid.
    Only check constraints involving new_elem:
    1. new_elem doesn't divide sum of any 3 from existing
    2. No existing element e divides (new_elem + x + y) for x,y in existing, x!=y, x!=e, y!=e
    """
    a = new_elem
    n = len(existing_list)
    
    # Check 1: a doesn't divide sum of any 3 from existing
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = existing_list[i] + existing_list[j] + existing_list[k]
                if s % a == 0:
                    return False
    
    # Check 2: no existing element e divides (a + x + y)
    for idx_e, e in enumerate(existing_list):
        others = [v for idx, v in enumerate(existing_list) if idx != idx_e]
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                s = a + others[i] + others[j]
                if s % e == 0:
                    return False
    
    return True

def _check_swap(old_elem, new_elem, existing_list):
    """Check if replacing old_elem with new_elem in existing_list is valid.
    Only check constraints involving new_elem or affected by removal of old_elem:
    1. new_elem doesn't divide sum of any 3 from new_list (existing minus old)
    2. Each e in new_list doesn't divide (new_elem + x + y) for x,y in new_list, x!=y, x!=e, y!=e
    """
    a = new_elem
    new_list = [v for v in existing_list if v != old_elem]
    n = len(new_list)
    
    # Check 1: a doesn't divide sum of any 3 from new_list
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = new_list[i] + new_list[j] + new_list[k]
                if s % a == 0:
                    return False
    
    # Check 2: each e in new_list doesn't divide (a + x + y)
    for idx_e, e in enumerate(new_list):
        others = [v for idx, v in enumerate(new_list) if idx != idx_e]
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                s = a + others[i] + others[j]
                if s % e == 0:
                    return False
    
    return True

def _greedy_construct():
    """Greedy construction from high to low."""
    selected = []
    for x in range(192, 0, -1):
        if _check_new_element(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline, max_rounds=300):
    """Local search with swaps and additions to improve the set."""
    selected = list(selected)
    selected_set = set(selected)
    best = selected[:]
    best_size = len(selected)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding elements (highest first)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in selected_set:
                if _check_new_element(x, selected):
                    selected.append(x)
                    selected_set.add(x)
                    improved = True
                    if len(selected) > best_size:
                        best = selected[:]
                        best_size = len(selected)
        
        # Try swapping elements
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            old = selected[i]
            for x in range(192, 0, -1):
                if x in selected_set:
                    continue
                if _check_swap(old, x, selected):
                    selected[i] = x
                    selected_set.discard(old)
                    selected_set.add(x)
                    improved = True
                    if len(selected) > best_size:
                        best = selected[:]
                        best_size = len(selected)
                    break
        
        if not improved:
            # Perturbation: remove a random element
            if len(selected) > 1:
                idx = random.randint(0, len(selected) - 1)
                removed = selected.pop(idx)
                selected_set.discard(removed)
                # Try to fill back in
                for x in range(192, 0, -1):
                    if time.time() > deadline:
                        break
                    if x not in selected_set:
                        if _check_new_element(x, selected):
                            selected.append(x)
                            selected_set.add(x)
                            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Initial greedy
    selected = _greedy_construct()
    best = selected[:]
    
    # Local search
    selected = _local_search(selected, deadline)
    if len(selected) > len(best):
        best = selected
    
    return best
# EVOLVE-BLOCK-END
