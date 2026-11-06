import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    n = len(arr)
    # Check subsets in decreasing order of size
    # For n=12, this is fast.
    for k in range(n, 0, -1):
        for combo in combinations(range(n), k):
            subset = [arr[i] for i in combo]
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # We want to minimize the max Sidon subset size.
    # Search in a small range to force collisions.
    # Range [0, 25] is a good candidate based on knowledge.
    search_range = 26 
    max_val = 128
    
    best_set = None
    best_val = 13 # Higher than max possible (12)
    
    # We try to construct a set where the max Sidon size is <= target_size
    # We iterate target_size from 4 upwards. If we find a valid set with size 4, we stop.
    # If not, we try 5, etc.
    
    for target_size in range(4, 13):
        if time.time() > deadline:
            break
            
        # Greedy construction
        current_set = []
        
        for x in range(search_range):
            if len(current_set) == 12:
                break
                
            # Check if adding x is valid with respect to target_size
            # Note: max_sidon_size is expensive, but for small n it's okay.
            # We can optimize by checking if the new element creates a conflict 
            # that forces a larger Sidon set, but simple check is robust.
            
            candidate_set = current_set + [x]
            
            # Optimization: If current_set already has a Sidon subset of size target_size,
            # adding x might increase it or keep it. 
            # Actually, max_sidon_size(candidate_set) >= max_sidon_size(current_set).
            # If max_sidon_size(current_set) > target_size, we would have failed earlier.
            
            val = max_sidon_size(candidate_set)
            
            if val <= target_size:
                current_set.append(x)
                
        if len(current_set) == 12:
            final_val = max_sidon_size(current_set)
            
            # If we found a set with size <= target_size, it's good.
            # We prefer lower values.
            if final_val < best_val:
                best_val = final_val
                best_set = current_set[:]
                
            # If we achieved the target, we can try to polish or just return.
            # Since we iterate target_size from 4 up, the first success with 4 is optimal.
            # If we succeed with 5, but 4 was impossible, 5 is the best we can do with this greedy.
            # Let's do a quick local search to see if we can drop to target_size - 1.
            
            # Local Search Polish
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(12):
                    if time.time() > deadline:
                        break
                    old_val = current_set[i]
                    for new_val in range(search_range):
                        if new_val == old_val:
                            continue
                        if new_val in current_set:
                            continue
                        
                        current_set[i] = new_val
                        new_size = max_sidon_size(current_set)
                        
                        if new_size < final_val:
                            final_val = new_size
                            improved = True
                            break
                        else:
                            current_set[i] = old_val
                    
                    if improved:
                        break
            
            if final_val < best_val:
                best_val = final_val
                best_set = current_set[:]
                
            # If we found a set with size 4, we can't do better (min possible is 1? 
            # Actually, min Sidon size for 12 elements is likely 4 or 5. 
            # A Sidon set of size 1 is trivial. 
            # The metric is the MAX sidon subset size.
            # Can we get 3? 
            # If max sidon size is 3, it means no 4 distinct numbers form a Sidon set.
            # This is a very strong constraint.
            # Let's see if we can get 4.
            
            if best_val <= 4:
                return best_set
                
    # Fallback: if greedy failed or didn't find good ones, try random search in small range
    if best_set is None:
        import random
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            s = random.sample(range(search_range), 12)
            v = max_sidon_size(s)
            if v < best_val:
                best_val = v
                best_set = s
                if v <= 4:
                    break

    if best_set is None:
        # Absolute fallback
        best_set = list(range(12))
        
    return best_set
