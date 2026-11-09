import random
import time
import sys

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    def _is_valid_add(x, chosen, counts):
        """Check if x can be added to chosen without violating B2[2]."""
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
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
    
    def _greedy_init(order=None):
        """Initialize with a greedy set using given order (default ascending)."""
        if order is None:
            order = list(range(1, 1001))
        chosen = []
        counts = {}
        for x in order:
            if _is_valid_add(x, chosen, counts):
                _add_element(x, chosen, counts)
        return chosen, counts
    
    def _local_search(current_set, current_counts, best_size, timeout_left):
        """Perform local search to improve the set."""
        best_set = current_set[:]
        best_counts = current_counts.copy()
        best_sz = len(best_set)
        
        # Build a fast lookup for membership
        in_set = set(current_set)
        
        while time.time() - start_time < timeout:
            improved = False
            
            # Phase 1: Try to add any element
            candidates = [x for x in range(1, 1001) if x not in in_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if _is_valid_add(x, current_set, current_counts):
                    _add_element(x, current_set, current_counts)
                    in_set.add(x)
                    improved = True
                    if len(current_set) > best_sz:
                        best_set = current_set[:]
                        best_counts = current_counts.copy()
                        best_sz = len(best_set)
                    break
            
            if not improved:
                # Phase 2: Try swaps
                elems = current_set[:]
                random.shuffle(elems)
                
                for u in elems:
                    _remove_element(u, current_set, current_counts)
                    in_set.discard(u)
                    
                    candidates = [x for x in range(1, 1001) if x not in in_set]
                    random.shuffle(candidates)
                    
                    swapped = False
                    for x in candidates:
                        if _is_valid_add(x, current_set, current_counts):
                            _add_element(x, current_set, current_counts)
                            in_set.add(x)
                            swapped = True
                            break
                    
                    if not swapped:
                        # Rollback
                        _add_element(u, current_set, current_counts)
                        in_set.add(u)
                    else:
                        improved = True
                        break
            
            if not improved:
                # Stuck - try a kick or restart
                if time.time() - start_time > timeout - 5:
                    break
                
                # Random restart
                order = list(range(1, 1001))
                random.shuffle(order)
                current_set, current_counts = _greedy_init(order)
                in_set = set(current_set)
                if len(current_set) > best_sz:
                    best_set = current_set[:]
                    best_counts = current_counts.copy()
                    best_sz = len(best_set)
        
        return best_set, best_counts, best_sz
    
    # 1. Initial greedy
    best_set, best_counts = _greedy_init()
    best_size = len(best_set)
    
    # 2. Local search
    current_set, current_counts = best_set[:], best_counts.copy()
    best_set, best_counts, best_size = _local_search(current_set, current_counts, best_size, timeout)
    
    # 3. Try a few more random restarts if time permits
    attempts = 0
    while time.time() - start_time < timeout - 5 and attempts < 10:
        order = list(range(1, 1001))
        random.shuffle(order)
        current_set, current_counts = _greedy_init(order)
        
        if len(current_set) > best_size:
            best_set, best_counts, best_size = _local_search(current_set, current_counts, best_size, timeout)
        else:
            # Even if not better, try local search - might find improvement
            current_set, current_counts, local_size = _local_search(current_set, current_counts, best_size, timeout)
            if local_size > best_size:
                best_set, best_counts, best_size = current_set, current_counts, local_size
        
        attempts += 1
    
    return best_set
