import random
import time

def _is_valid_add(x, chosen, counts):
    """Check if x can be added to chosen without violating B2[2]."""
    # x + y for y in chosen
    for y in chosen:
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
    # 2x
    if counts.get(2 * x, 0) >= 2:
        return False
    return True

def _add_element(x, chosen, counts):
    """Add x to chosen and update counts."""
    for y in chosen:
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    counts[2 * x] = counts.get(2 * x, 0) + 1
    chosen.append(x)

def _remove_element(x, chosen, counts):
    """Remove x from chosen and update counts."""
    for i in range(len(chosen) - 1, -1, -1):
        if chosen[i] == x:
            chosen.pop(i)
            break
    for y in chosen:
        s = x + y
        counts[s] -= 1
    counts[2 * x] -= 1

def _greedy_init():
    """Initialize with a greedy ascending set."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        if _is_valid_add(x, chosen, counts):
            _add_element(x, chosen, counts)
    return chosen, counts

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    # 1. Initialize
    best_set, best_counts = _greedy_init()
    best_size = len(best_set)
    
    # 2. Local Search
    # We will try to improve the set by adding elements or swapping.
    # We keep the current state and try to beat the best found.
    
    current_set, current_counts = best_set[:], best_counts.copy()
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Phase 1: Try to add any element
        candidates = [x for x in range(1, 1001) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if _is_valid_add(x, current_set, current_counts):
                _add_element(x, current_set, current_counts)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set[:]
                    best_counts = current_counts.copy()
                    best_size = len(best_set)
                break # Restart candidate search after an addition
                
        if not improved:
            # Phase 2: Try swaps (remove one, add one)
            # This allows escaping local minima where no single addition is possible
            # but a swap might open up space.
            
            # Create a list of elements to try removing
            elems = current_set[:]
            random.shuffle(elems)
            
            for u in elems:
                # Try removing u
                _remove_element(u, current_set, current_counts)
                
                # Try to add some x
                # We can check candidates more efficiently here
                # But for simplicity, shuffle all not in set
                # Note: u is not in current_set now
                candidates = [x for x in range(1, 1001) if x not in current_set]
                random.shuffle(candidates)
                
                swapped = False
                for x in candidates:
                    if _is_valid_add(x, current_set, current_counts):
                        _add_element(x, current_set, current_counts)
                        swapped = True
                        # Size is same, so best_size doesn't change, 
                        # but the configuration might allow further additions later.
                        # We update best_set only if size increases, 
                        # but we should keep the current configuration for further search.
                        break
                
                if not swapped:
                    # Rollback: re-add u
                    _add_element(u, current_set, current_counts)
                else:
                    # We performed a swap. 
                    # Should we continue searching for more additions immediately?
                    # Yes, the loop will go back to Phase 1.
                    improved = True
                    break # Break out of removal loop to restart Phase 1
        
        if not improved:
            # If we couldn't add or swap, we are stuck.
            # Try a random restart from a slightly perturbed state?
            # Or just break if we've been stuck for a while?
            # Let's try a few random restarts of the local search if stuck.
            # To save code complexity, let's just restart the local search process
            # by randomizing the current set a bit?
            # Actually, if we are stuck, it means no single addition or swap works.
            # We can try a "kick": remove a random element, then try to add multiple?
            # Or just start a new greedy init with random shuffle?
            
            # Let's do a random restart of the entire local search process
            # to avoid getting trapped in a deep local minimum.
            # We'll re-initialize with a random greedy order.
            
            # Check time
            if time.time() - start_time > timeout - 5:
                break
                
            # Random Greedy Init
            order = list(range(1, 1001))
            random.shuffle(order)
            current_set = []
            current_counts = {}
            for x in order:
                if _is_valid_add(x, current_set, current_counts):
                    _add_element(x, current_set, current_counts)
                    
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_counts = current_counts.copy()
                best_size = len(current_set)
                
            # Continue loop to try to improve this new state

    return best_set
