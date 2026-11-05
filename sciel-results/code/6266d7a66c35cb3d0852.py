import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 60s limit, leave 5s margin
    
    all_elements = list(range(1, 101))
    
    # Helper to check validity using bitset with early exit
    def check_validity(s):
        r = 1
        for x in s:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True

    def greedy_init(order):
        selected = []
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not (new_reach >> 200) & 1:
                selected.append(x)
                reachable = new_reach
        return set(selected)

    # 1. Multi-start Greedy Initialization
    best_set = set()
    best_size = 0
    
    # Descending
    s1 = greedy_init(range(100, 0, -1))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # Ascending
    s2 = greedy_init(range(1, 101))
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # Randomized Greedy (10 trials)
    for _ in range(10):
        random_order = all_elements.copy()
        random.shuffle(random_order)
        s3 = greedy_init(random_order)
        if len(s3) > best_size:
            best_set = s3
            best_size = len(s3)

    current_set = best_set.copy()
    current_size = best_size

    # 2. Systematic Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Step 1: Try to add any element not in current_set (0-for-1)
        # We iterate through all elements to find one that can be added
        for x in all_elements:
            if x not in current_set:
                new_set = current_set | {x}
                if check_validity(new_set):
                    current_set = new_set
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_set = current_set.copy()
                        best_size = current_size
                    break # Restart loop to try adding more
        
        if time.time() > deadline:
            break
            
        # Step 2: Try 1-for-2 swaps
        # We look for a move that increases size by 1
        candidates = [x for x in all_elements if x not in current_set]
        
        if len(candidates) < 2:
            break
            
        found_swap = False
        # Iterate through elements to remove
        for x_remove in list(current_set):
            if time.time() > deadline:
                break
            if found_swap:
                break
                
            # Try all pairs from candidates
            # Optimization: We can break early if found
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                y = candidates[i]
                for j in range(i + 1, len(candidates)):
                    z = candidates[j]
                    # Construct new set efficiently
                    new_set = current_set.copy()
                    new_set.discard(x_remove)
                    new_set.add(y)
                    new_set.add(z)
                    
                    if check_validity(new_set):
                        current_set = new_set
                        current_size += 1
                        if current_size > best_size:
                            best_set = current_set.copy()
                            best_size = current_size
                        improved = True
                        found_swap = True
                        break

    return list(best_set)
