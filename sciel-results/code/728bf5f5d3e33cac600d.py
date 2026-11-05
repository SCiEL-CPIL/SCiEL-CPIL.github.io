<<<<<<< SEARCH
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy pass (deterministic)
    chosen, counts = [], {}
    for x in range(1, 1001):
        valid = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            s = 2 * x
            if counts.get(s, 0) + 1 > 2:
                valid = False
        if valid:
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
    
    best_set = chosen
    best_size = len(chosen)
    
    # Randomized greedy with restarts
    for _ in range(50):  # Limit restarts to fit in time
        if time.time() > deadline:
            break
            
        order = random.sample(range(1, 1001), 1000)
        chosen, counts = [], {}
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            if valid:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen
    
    return best_set
=======
import time
import random

def _valid_add(chosen_set, x, counts):
    """Check if x can be added to chosen_set without violating B2[2]."""
    for y in chosen_set:
        s = x + y
        if counts.get(s, 0) + 1 > 2:
            return False
    s = 2 * x
    if counts.get(s, 0) + 1 > 2:
        return False
    return True

def _add_to_set(chosen_set, x, counts):
    """Add x to chosen_set and update counts."""
    for y in chosen_set:
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    s = 2 * x
    counts[s] = counts.get(s, 0) + 1
    chosen_set.append(x)

def _remove_from_set(chosen_set, x, counts):
    """Remove x from chosen_set and update counts."""
    for y in chosen_set:
        if y == x:
            continue
        s = x + y
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
    s = 2 * x
    counts[s] -= 1
    if counts[s] == 0:
        del counts[s]
    chosen_set.remove(x)

def _greedy_pass(order, time_limit=None):
    """Run greedy pass over given order."""
    chosen, counts = [], {}
    for x in order:
        if time_limit is not None and time.time() > time_limit:
            break
        if _valid_add(chosen, x, counts):
            _add_to_set(chosen, x, counts)
    return chosen, counts

def _local_improve(chosen, counts, max_attempts=5000, time_limit=None):
    """
    Local improvement via swap search:
    - Try to add elements not in the set
    - Try to swap: remove one element, add another
    """
    chosen_set = list(chosen)
    counts = dict(counts)
    n = len(chosen_set)
    chosen_in_set = set(chosen_set)
    
    improved = True
    attempts = 0
    while improved and attempts < max_attempts:
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False
        attempts += 1
        
        # Phase 1: Try to add any element not in set
        for x in range(1, 1001):
            if x in chosen_in_set:
                continue
            if _valid_add(chosen_set, x, counts):
                _add_to_set(chosen_set, x, counts)
                chosen_in_set.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps (remove one, add another)
        swapped = False
        for i in range(len(chosen_set)):
            if time_limit is not None and time.time() > time_limit:
                break
            if swapped:
                break
            x_remove = chosen_set[i]
            # Temporarily remove
            _remove_from_set(chosen_set, x_remove, counts)
            chosen_in_set.discard(x_remove)
            
            for x_add in range(1, 1001):
                if x_add in chosen_in_set:
                    continue
                if _valid_add(chosen_set, x_add, counts):
                    _add_to_set(chosen_set, x_add, counts)
                    chosen_in_set.add(x_add)
                    swapped = True
                    improved = True
                    break
            
            if not swapped:
                # Re-add x_remove
                _add_to_set(chosen_set, x_remove, counts)
                chosen_in_set.add(x_remove)
    
    return chosen_set, counts

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy pass (deterministic)
    order = list(range(1, 1001))
    chosen, counts = _greedy_pass(order)
    best_set = list(chosen)
    best_size = len(chosen)
    
    # Randomized greedy with restarts + local improvement
    for restart in range(80):
        if time.time() > deadline:
            break
        
        order = random.sample(range(1, 1001), 1000)
        chosen, counts = _greedy_pass(order)
        
        # Local improvement
        chosen, counts = _local_improve(chosen, counts, max_attempts=2000, time_limit=deadline)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
    
    # Final local improvement on best set
    if time.time() < deadline - 5:
        best_set, _ = _local_improve(best_set, {}, max_attempts=5000, time_limit=deadline)
        # Rebuild counts for best_set
        counts = {}
        for i, a in enumerate(best_set):
            for j in range(i, len(best_set)):
                s = a + best_set[j]
                counts[s] = counts.get(s, 0) + 1
        best_set, _ = _local_improve(best_set, counts, max_attempts=3000, time_limit=deadline)
    
    return best_set
>>>>>>> REPLACE
