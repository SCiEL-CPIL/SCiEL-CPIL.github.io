import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 41))
    
    def is_valid_subset(subset):
        """Check if all subset products are distinct."""
        products = {1}
        for c in subset:
            new_products = set()
            for p in products:
                np = p * c
                if np in products:
                    return False
                new_products.add(np)
            products.update(new_products)
        return True
    
    # Try greedy with different orderings
    best_size = 0
    best_set = []
    
    # Ordering 1: increasing
    orderings = [
        list(range(2, 41)),
        list(range(40, 1, -1)),
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40],
        [37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2, 36, 35, 34, 33, 32, 30, 28, 27, 26, 25, 24, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 4],
    ]
    
    for order in orderings:
        selected = []
        products = {1}
        
        for c in order:
            new_products = set()
            valid = True
            for p in products:
                np = p * c
                if np in products:
                    valid = False
                    break
                new_products.add(np)
            
            if valid:
                selected.append(c)
                products.update(new_products)
            
            if time.time() > deadline:
                break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        
        if time.time() > deadline:
            break
    
    # Now try to improve with local search / backtracking
    if time.time() < deadline - 5:
        # Try to add more by removing some and replacing
        current = set(best_set)
        remaining = [c for c in range(2, 41) if c not in current]
        
        # Try each remaining number, see if we can add it by removing one current number
        for r in remaining:
            for c in list(current):
                trial = (current - {c}) | {r}
                if is_valid_subset(list(trial)):
                    # We have same size, but maybe better for further additions
                    pass
        
        # Try to find a larger set via targeted search
        # Use the current best as a starting point and try to extend
        if len(best_set) < 12:
            # Try DFS with pruning to find a larger set
            best_size = len(best_set)
            
            def dfs(idx, selected, products):
                nonlocal best_size, best_set, deadline, start_time
                
                if time.time() > deadline - 2:
                    return
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = list(selected)
                
                if idx >= len(candidates):
                    return
                
                # Pruning: even if we take all remaining, can we beat best?
                if len(selected) + (len(candidates) - idx) <= best_size:
                    return
                
                c = candidates[idx]
                
                # Option 1: include c
                new_products = set()
                valid = True
                for p in products:
                    np = p * c
                    if np in products:
                        valid = False
                        break
                    new_products.add(np)
                
                if valid:
                    dfs(idx + 1, selected + [c], products | new_products)
                
                # Option 2: exclude c
                dfs(idx + 1, selected, products)
            
            # Start DFS from empty set with a good ordering
            # Use the ordering that worked best
            dfs_order = orderings[0]  # increasing
            dfs(0, [], {1})
    
    return [int(x) for x in best_set]
