# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    best_set = []
    best_size = 0

    def _greedy_numpy(order, counts_arr):
        """Greedy construction using numpy array for fast sum counting."""
        chosen = []
        for x in order:
            if counts_arr[2 * x] >= 2:
                continue
            valid = True
            for y in chosen:
                s = x + y
                if counts_arr[s] >= 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts_arr[2 * x] += 1
                for y in chosen[:-1]:
                    s = x + y
                    counts_arr[s] += 1
        return chosen

    def _can_add(x, chosen, counts_arr, chosen_set=None):
        """Check if x can be added to chosen without violating B2[2]."""
        if x in (chosen_set if chosen_set is not None else chosen):
            return False
        if counts_arr[2 * x] >= 2:
            return False
        for y in chosen:
            if counts_arr[x + y] >= 2:
                return False
        return True

    def _add(x, chosen, counts_arr, chosen_set=None):
        """Add x to chosen and update counts."""
        chosen.append(x)
        if chosen_set is not None:
            chosen_set.add(x)
        counts_arr[2 * x] += 1
        for y in chosen[:-1]:
            counts_arr[x + y] += 1

    def _remove(x, chosen, counts_arr, chosen_set=None):
        """Remove x from chosen and update counts."""
        idx = chosen.index(x)
        chosen.pop(idx)
        if chosen_set is not None:
            chosen_set.discard(x)
        counts_arr[2 * x] -= 1
        for y in chosen:
            counts_arr[x + y] -= 1

    def _try_add_any(chosen, counts_arr, chosen_set):
        """Try to add any element from 1..1000. Returns True if added."""
        for x in range(1, 1001):
            if _can_add(x, chosen, counts_arr, chosen_set):
                _add(x, chosen, counts_arr, chosen_set)
                return True
        return False

    def _local_search_improve(chosen, time_limit):
        """Improved local search with vectorized feasibility and drop-repair."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts_arr = np.zeros(2001, dtype=np.int32)
        
        for i, a in enumerate(chosen):
            counts_arr[2 * a] += 1
            for b in chosen[i+1:]:
                counts_arr[a + b] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(chosen)
            
            # Phase 1: Try adding a new element
            if _try_add_any(chosen, counts_arr, chosen_set):
                improved = True
                continue
            
            # Phase 2: Try single swaps (remove one, add one)
            # Use random order to escape local minima
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= time_limit:
                    break
                old = chosen[i]
                _remove(old, chosen, counts_arr, chosen_set)
                
                # Try to add a new element
                added = False
                for x in range(1, 1001):
                    if x == old:
                        continue
                    if _can_add(x, chosen, counts_arr, chosen_set):
                        _add(x, chosen, counts_arr, chosen_set)
                        added = True
                        break
                
                if not added:
                    # Restore old
                    _add(old, chosen, counts_arr, chosen_set)
                else:
                    improved = True
                    break
            
            # Phase 3: Drop-and-repair (remove 2, try to add 3)
            if not improved and n >= 2 and time.time() < time_limit:
                # Remove two random elements
                i, j = random.sample(range(len(chosen)), 2)
                j = j if j > i else j  # keep as is
                old1 = chosen[i]
                old2 = chosen[j]
                # Remove in order of index (higher first to keep indices valid)
                hi, lo = max(i, j), min(i, j)
                _remove(chosen[hi], chosen, counts_arr, chosen_set)
                _remove(chosen[lo], chosen, counts_arr, chosen_set)
                
                # Try to add up to 3 new elements
                added_count = 0
                for _ in range(3):
                    if _try_add_any(chosen, counts_arr, chosen_set):
                        added_count += 1
                    else:
                        break
                
                if added_count < 2:
                    # Not enough added, restore
                    _add(old1, chosen, counts_arr, chosen_set)
                    _add(old2, chosen, counts_arr, chosen_set)
                else:
                    improved = True
        
        return chosen

    # Strategy 1: Forward greedy (small to large)
    order1 = list(range(1, 1001))
    counts1 = np.zeros(2001, dtype=np.int32)
    set1 = _greedy_numpy(order1, counts1)
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    # Strategy 2: Backward greedy (large to small)
    order2 = list(range(1000, 0, -1))
    counts2 = np.zeros(2001, dtype=np.int32)
    set2 = _greedy_numpy(order2, counts2)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    # Strategy 3: Random restarts with local search
    random.seed(42)
    while time.time() < deadline - 30:
        order = list(range(1, 1001))
        random.shuffle(order)
        
        counts = np.zeros(2001, dtype=np.int32)
        current_set = _greedy_numpy(order, counts)
        
        if time.time() < deadline - 10:
            local_time_limit = min(time.time() + 5, deadline - 5)
            current_set = _local_search_improve(current_set, local_time_limit)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Final local search on best set
    if time.time() < deadline - 5:
        best_set = _local_search_improve(best_set, deadline - 2)
        best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END
