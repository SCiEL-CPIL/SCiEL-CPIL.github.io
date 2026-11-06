# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    best_set = []
    best_size = 0

    def _can_add(x, chosen_arr, counts_arr):
        """Vectorized check if x can be added to the set."""
        if counts_arr[2 * x] >= 2:
            return False
        if len(chosen_arr) == 0:
            return True
        sums = x + chosen_arr
        return np.all(counts_arr[sums] < 2)

    def _greedy_numpy(order, counts_arr):
        """Greedy construction using numpy array for fast sum counting."""
        chosen = []
        chosen_set = set()
        for x in order:
            if x in chosen_set:
                continue
            if counts_arr[2 * x] >= 2:
                continue
            if len(chosen) > 0:
                sums = x + np.array(chosen)
                if np.any(counts_arr[sums] >= 2):
                    continue
            chosen.append(x)
            chosen_set.add(x)
            counts_arr[2 * x] += 1
            if len(chosen) > 1:
                for y in chosen[:-1]:
                    counts_arr[x + y] += 1
        return chosen

    def _local_search_improve(chosen, time_limit):
        """Try to improve the set by adding elements or swapping."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts_arr = np.zeros(2001, dtype=np.int32)
        
        # Rebuild counts
        for i, a in enumerate(chosen):
            counts_arr[2 * a] += 1
            for b in chosen[i+1:]:
                counts_arr[a + b] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(chosen)
            
            # Try adding a new element - vectorized
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                if _can_add(x, np.array(chosen) if chosen else np.array([]), counts_arr):
                    chosen.append(x)
                    chosen_set.add(x)
                    counts_arr[2 * x] += 1
                    if n > 0:
                        for y in chosen[:-1]:
                            counts_arr[x + y] += 1
                    improved = True
                    break
            
            if not improved and n > 0:
                # Try swapping: remove one element, add another
                for i in range(n):
                    old = chosen[i]
                    # Remove old from counts
                    counts_arr[2 * old] -= 1
                    for j, y in enumerate(chosen):
                        if j != i:
                            counts_arr[old + y] -= 1
                    
                    # Build array of remaining elements for vectorized check
                    remaining = [chosen[j] for j in range(n) if j != i]
                    remaining_arr = np.array(remaining) if remaining else np.array([])
                    
                    # Try to add a new element
                    found = False
                    for x in range(1, 1001):
                        if x == old or x in chosen_set:
                            continue
                        if _can_add(x, remaining_arr, counts_arr):
                            chosen[i] = x
                            chosen_set.discard(old)
                            chosen_set.add(x)
                            counts_arr[2 * x] += 1
                            for y in remaining:
                                counts_arr[x + y] += 1
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Restore old
                        counts_arr[2 * old] += 1
                        for y in remaining:
                            counts_arr[old + y] += 1
                        chosen_set.add(old)
                    else:
                        break
        
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
    while time.time() < deadline - 20:  # Leave 20s for final local search
        # Random permutation
        order = list(range(1, 1001))
        random.shuffle(order)
        
        counts = np.zeros(2001, dtype=np.int32)
        current_set = _greedy_numpy(order, counts)
        
        # Apply local search if time permits
        if time.time() < deadline - 8:
            local_time_limit = min(time.time() + 3, deadline - 5)
            current_set = _local_search_improve(current_set, local_time_limit)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Final local search on best set
    if time.time() < deadline - 3:
        best_set = _local_search_improve(best_set, deadline - 1)
        best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END
