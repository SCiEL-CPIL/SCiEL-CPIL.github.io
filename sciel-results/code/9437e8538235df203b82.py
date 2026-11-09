import time
import random

def construct():
    start_time = time.time()
    # Evaluator kills at 60s. Use 50s to be safe.
    time_limit = 50.0
    
    N = 512
    
    # Precompute 4-APs
    # An AP is (a, b, c, d) with b=a+d, c=a+2d, d=a+3d
    # Indices 0..N-1
    # Instead of storing full APs, store for each index x, a list of tuples (y, z, w)
    # such that {x, y, z, w} forms a 4-AP.
    # This allows O(1) check: if in_set[y] and in_set[z] and in_set[w] then conflict.
    
    aps_by_element = [[] for _ in range(N)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For a: others are b, c, e
            aps_by_element[a].append((b, c, e))
            # For b: others are a, c, e
            aps_by_element[b].append((a, c, e))
            # For c: others are a, b, e
            aps_by_element[c].append((a, b, e))
            # For e: others are a, b, c
            aps_by_element[e].append((a, b, c))
            
    # Helper to check if x can be added to in_set (bytearray of 0/1)
    def can_add(x, in_set):
        for y, z, w in aps_by_element[x]:
            if in_set[y] and in_set[z] and in_set[w]:
                return False
        return True

    # Precompute constraint counts for smarter greedy
    constraint_counts = [len(aps) for aps in aps_by_element]
    
    def greedy_init():
        in_set = bytearray(N)
        elements = list(range(N))
        random.shuffle(elements)  
        elements.sort(key=lambda x: constraint_counts[x] + random.random() * 0.1)
        count = 0
        for x in elements:
            if can_add(x, in_set):
                in_set[x] = 1
                count += 1
        return in_set, count

    def local_search(in_set, count):
        # Try to add elements until no more can be added
        # We maintain a list of missing elements.
        missing = [i for i in range(N) if not in_set[i]]
        random.shuffle(missing)
        
        # Pass 1: Add any possible missing elements
        # Since adding elements only restricts future additions, we only need one pass.
        new_missing = []
        for x in missing:
            if can_add(x, in_set):
                in_set[x] = 1
                count += 1
            else:
                new_missing.append(x)
        missing = new_missing
        
        # Swap-based local search: try removing one element to add two others
        current_elements = [i for i in range(N) if in_set[i]]
        random.shuffle(current_elements)
        random.shuffle(missing)
        
        # We will iterate until no improvement or time limit
        improved = True
        while improved:
            improved = False
            for rem in current_elements:
                if time.time() - start_time > time_limit:
                    return in_set, count
                
                # Try removing rem
                in_set[rem] = 0
                count -= 1
                
                # Try adding two missing elements
                added = []
                # We iterate through missing list. 
                # Note: If we add an element, it might block others in `missing`.
                # But since we only add up to 2, and we check `can_add` dynamically, it's correct.
                # However, `can_add` checks against current `in_set`.
                for add in missing:
                    if len(added) >= 2:
                        break
                    if can_add(add, in_set):
                        in_set[add] = 1
                        count += 1
                        added.append(add)
                
                # If we added at least 2, keep the change (net gain >= 1)
                if len(added) >= 2:
                    # Update lists
                    # Remove rem from current_elements
                    # Add added to current_elements
                    # Remove added from missing
                    # Add rem to missing
                    
                    # Optimization: Instead of list remove (O(N)), we can just mark and clean up later?
                    # Given N=512, list remove is fast enough (512 ops).
                    current_elements.remove(rem)
                    for a in added:
                        current_elements.append(a)
                    for a in added:
                        missing.remove(a)
                    missing.append(rem)
                    improved = True
                else:
                    # Revert
                    in_set[rem] = 1
                    count += 1
                    for a in added:
                        in_set[a] = 0
                        count -= 1
                        missing.append(a)
                    # current_elements and missing remain unchanged logically, 
                    # but we modified in_set. 
                    # Wait, we didn't modify current_elements/missing in revert case?
                    # Correct, we just reverted in_set.
                
                if time.time() - start_time > time_limit:
                    return in_set, count
        
        return in_set, count

    def perturb(in_set, count, k):
        current_elements = [i for i in range(N) if in_set[i]]
        random.shuffle(current_elements)
        for i in range(min(k, len(current_elements))):
            in_set[current_elements[i]] = 0
            count -= 1
        return in_set, count

    best_count = 0
    best_in_set = bytearray(N)
    
    # Initial random greedy
    in_set, count = greedy_init()
    in_set, count = local_search(in_set, count)
    if count > best_count:
        best_count = count
        best_in_set = in_set.copy()
        
    while time.time() - start_time < time_limit:
        # Perturb the current best or a random state?
        if random.random() < 0.5 and best_count > 0:
            in_set = best_in_set.copy()
            count = best_count
            in_set, count = perturb(in_set, count, 5)
        else:
            in_set, count = greedy_init()
        
        in_set, count = local_search(in_set, count)
        
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
            
    # Final polish: try more aggressive swaps on the best solution
    # Re-use local_search on the best solution found
    in_set = best_in_set.copy()
    count = best_count
    in_set, count = local_search(in_set, count)
    
    if count > best_count:
        best_count = count
        best_in_set = in_set.copy()
    
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if best_in_set[i]]
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result
