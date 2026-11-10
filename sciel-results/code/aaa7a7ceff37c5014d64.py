import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    max_val_limit = 1048576
    
    # Greedy construction with backtracking
    # We want to find 12 distinct positive integers with all subset sums unique
    # and minimize the largest element
    
    # Start with a known good solution: powers of 2
    best_arr = [1 << i for i in range(n)]
    best_max = max(best_arr)
    
    # Try greedy construction: pick smallest possible value at each step
    # that doesn't create duplicate subset sums
    
    def _build_greedy(max_element_limit, start_index=0, current_arr=None, current_sums=None):
        """
        Build array greedily, trying smallest values first.
        Returns (arr, sums) if successful, None if failed.
        """
        if current_arr is None:
            current_arr = []
            current_sums = {0}
        
        if len(current_arr) == n:
            return (current_arr[:], current_sums)
        
        # Try values from 1 upward (but must be distinct from existing)
        existing_values = set(current_arr)
        
        for val in range(1, max_element_limit + 1):
            if val in existing_values:
                continue
            
            # Check if adding val creates any duplicate subset sums
            # New sums would be {s + val for s in current_sums}
            # We need: no element in new_sums already in current_sums
            # AND no duplicates within new_sums (but since val > 0 and current_sums are distinct, new_sums will be distinct)
            # Actually we just need: (current_sums ∪ (current_sums + val)) has size = 2 * len(current_sums)
            
            new_sums_set = set()
            has_collision = False
            for s in current_sums:
                ns = s + val
                if ns in current_sums or ns in new_sums_set:
                    has_collision = True
                    break
                new_sums_set.add(ns)
            
            if not has_collision:
                # Try this value
                new_arr = current_arr + [val]
                new_sums = current_sums | new_sums_set
                
                # Check time
                if time.time() > deadline:
                    return None
                
                # Recurse
                result = _build_greedy(max_element_limit, start_index, new_arr, new_sums)
                if result is not None:
                    return result
        
        return None
    
    # First, try to find a solution with small max element using greedy
    # Start with a reasonable upper bound and try to minimize
    
    # The theoretical minimum for n elements with distinct subset sums is related to binary representation
    # For n=12, powers of 2 give max=2048, but we can do better
    
    # Try greedy with increasing upper bounds
    # Start with a bound that's likely to work
    upper_bound = 100  # Start small, increase if needed
    
    found = None
    while upper_bound <= max_val_limit and time.time() < deadline:
        result = _build_greedy(upper_bound)
        if result is not None:
            arr, sums = result
            if max(arr) < best_max:
                best_arr = arr
                best_max = max(arr)
            found = arr
            break
        upper_bound *= 2
    
    if found is None:
        # Fallback to powers of 2
        best_arr = [1 << i for i in range(n)]
        best_max = max(best_arr)
    
    # Now try to improve using local search
    current_arr = best_arr[:]
    current_max = best_max
    
    T = 50.0
    T_min = 0.1
    alpha = 0.995
    
    def _subset_sums(arr):
        sums = set()
        sums.add(0)
        for x in arr:
            new_sums = {s + x for s in sums}
            sums.update(new_sums)
        return sums
    
    while time.time() < deadline:
        # Generate neighbor
        neighbor = current_arr[:]
        idx = random.randint(0, n - 1)
        # Try to decrease the value or swap
        if random.random() < 0.5:
            # Try smaller value
            old_val = neighbor[idx]
            new_val = max(1, old_val - random.randint(1, max(1, old_val // 4)))
            if new_val != old_val:
                neighbor[idx] = new_val
        else:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        # Check validity
        sums = _subset_sums(neighbor)
        is_valid = (len(sums) == (1 << n))
        
        if is_valid:
            new_max = max(neighbor)
            new_cost = new_max
        else:
            collisions = (1 << n) - len(sums)
            new_cost = max(neighbor) + collisions * 1000
        
        delta_cost = new_cost - current_max
        
        if delta_cost < 0:
            current_arr = neighbor
            current_max = new_cost
            if is_valid and new_max < best_max:
                best_max = new_max
                best_arr = neighbor[:]
        else:
            if T > T_min:
                prob = 2.71828 ** (-delta_cost / T)
                if random.random() < prob:
                    current_arr = neighbor
                    current_max = new_cost
        
        T *= alpha
    
    return best_arr
