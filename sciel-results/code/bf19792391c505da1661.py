import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Leave 15 seconds margin for safety
    
    all_numbers = list(range(2, 41))
    
    def compute_products(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                new_prods.add(x * p)
            prods = new_prods
        return prods
        
    def can_add(prods, x):
        """Check if x can be added to the set with current products prods."""
        for p in prods:
            if x * p in prods:
                return False, None
        new_prods = set(prods)
        for p in prods:
            new_prods.add(x * p)
        return True, new_prods

    def is_valid(s):
        prods = compute_products(s)
        return len(prods) == (1 << len(s))

    best_set = []
    best_size = 0
    
    # Known good set
    base_set = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    if is_valid(base_set):
        best_set = list(base_set)
        best_size = len(best_set)
        
    def dfs_search(candidates, current_set, current_prods, depth, max_depth):
        """DFS search trying to add more elements. Returns best set found."""
        nonlocal best_set, best_size
        if time.time() > deadline:
            return current_set
        
        # Update best if we found a larger set
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
        
        if depth >= max_depth:
            return current_set
        
        # Try adding each candidate in reverse order (larger first)
        for i in range(len(candidates) - 1, -1, -1):
            if time.time() > deadline:
                return current_set
            
            x = candidates[i]
            valid, new_prods = can_add(current_prods, x)
            if valid:
                new_set = current_set + [x]
                remaining = candidates[:i]  # Only consider smaller candidates
                dfs_search(remaining, new_set, new_prods, depth + 1, max_depth)
                
        return current_set
    
    # DFS search from empty set with all candidates sorted in reverse
    if time.time() < deadline:
        dfs_search(list(reversed(all_numbers)), [], {1}, 0, 18)
    
    # If DFS didn't find 18, try local search from the best found set
    if best_size < 18 and time.time() < deadline:
        def local_search(initial_set, max_iterations=500):
            """Local search with add and swap moves."""
            if time.time() > deadline:
                return initial_set
                
            current_set = list(initial_set)
            current_prods = compute_products(current_set)
            current_size = len(current_set)
            best_local = list(current_set)
            best_local_size = current_size
            
            all_set = set(range(2, 41))
            
            for _ in range(max_iterations):
                if time.time() > deadline:
                    break
                    
                improved = False
                current_set_set = set(current_set)
                
                # 1. Try to add (in reverse order)
                for x in sorted(all_set - current_set_set, reverse=True):
                    valid, new_prods = can_add(current_prods, x)
                    if valid:
                        current_set.append(x)
                        current_prods = new_prods
                        current_size += 1
                        improved = True
                        if current_size > best_local_size:
                            best_local = list(current_set)
                            best_local_size = current_size
                        break  # Restart after add
                        
                if improved:
                    continue
                    
                # 2. Try 1-swap
                swapped = False
                for i, elem_to_remove in enumerate(current_set):
                    if time.time() > deadline:
                        break
                    temp_set = current_set[:i] + current_set[i+1:]
                    temp_prods = compute_products(temp_set)
                    
                    for x in sorted(all_set - set(temp_set), reverse=True):
                        if x == elem_to_remove:
                            continue
                        valid, new_prods = can_add(temp_prods, x)
                        if valid:
                            current_set = temp_set + [x]
                            current_prods = new_prods
                            swapped = True
                            if len(current_set) > best_local_size:
                                best_local = list(current_set)
                                best_local_size = len(current_set)
                            break
                    if swapped:
                        break
                        
                if not improved and not swapped:
                    break
                    
            return best_local
        
        # Run local search from the best DFS result
        if time.time() < deadline:
            improved_set = local_search(best_set, max_iterations=300)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
                
        # Run local search from base set if better
        if time.time() < deadline:
            improved_set = local_search(base_set, max_iterations=300)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
    
    # Final verification
    best_set = [x for x in best_set if 2 <= x <= 40]
    best_set = list(set(best_set))
    best_set.sort()
    
    if is_valid(best_set):
        return best_set
    else:
        return list(base_set)

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
