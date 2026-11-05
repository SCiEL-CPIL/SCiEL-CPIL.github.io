import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    
    # Known good solution: {66, ..., 100}, size 35
    best_set = set(range(66, 101))
    best_size = 35
    
    def is_valid(s):
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True
    
    def compute_reachable(s):
        """Compute all subset sums up to 200 for set s."""
        reachable = {0}
        for x in s:
            new_sums = {y + x for y in reachable if y + x <= 200}
            reachable |= new_sums
        return reachable
    
    # Greedy construction with different orders
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 50, -1)) + list(range(50, 0, -1))
    ]
    
    # Add random orders
    for _ in range(15):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        s = set()
        reachable = {0}
        
        for x in order:
            if (200 - x) not in reachable:
                s.add(x)
                new_sums = {y + x for y in reachable if y + x <= 200}
                reachable |= new_sums
        
        if len(s) > best_size:
            if is_valid(s):
                best_size = len(s)
                best_set = s
    
    # Local search: try to improve by swapping or adding elements
    # Strategy: 
    # 1. Try to add a single element not in best_set
    # 2. Try to swap one element out and two in (net +1)
    # 3. Try to swap two out and three in (net +1)
    
    def try_local_search(initial_set, max_iterations=1000):
        """Try to improve the set via local search moves."""
        s = set(initial_set)
        reachable = compute_reachable(s)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            
            # Move 1: Add a single element
            for x in range(1, 101):
                if x not in s:
                    if (200 - x) not in reachable:
                        s.add(x)
                        new_sums = {y + x for y in reachable if y + x <= 200}
                        reachable |= new_sums
                        improved = True
                        break
            
            if improved:
                continue
            
            # Move 2: Swap one out, two in
            for x_out in list(s):
                if time.time() > deadline:
                    return s
                # Temporarily remove x_out
                s.remove(x_out)
                # Recompute reachable without x_out
                reachable = compute_reachable(s)
                
                # Try to add two elements
                added = []
                for x1 in range(1, 101):
                    if x1 not in s and x1 != x_out:
                        if (200 - x1) not in reachable:
                            s.add(x1)
                            new_sums = {y + x1 for y in reachable if y + x1 <= 200}
                            reachable |= new_sums
                            added.append(x1)
                            break
                
                if len(added) == 1:
                    for x2 in range(1, 101):
                        if x2 not in s and x2 != x_out and x2 != added[0]:
                            if (200 - x2) not in reachable:
                                s.add(x2)
                                new_sums = {y + x2 for y in reachable if y + x2 <= 200}
                                reachable |= new_sums
                                added.append(x2)
                                break
                
                if len(added) == 2:
                    improved = True
                    break
                else:
                    # Undo: remove what we added
                    for x_add in added:
                        s.discard(x_add)
                    # Recompute reachable
                    reachable = compute_reachable(s)
            
            if improved:
                continue
            
            # Move 3: Swap two out, three in
            for x1_out in list(s):
                if time.time() > deadline:
                    return s
                for x2_out in list(s):
                    if x2_out == x1_out:
                        continue
                    if time.time() > deadline:
                        return s
                    
                    # Temporarily remove both
                    s.remove(x1_out)
                    s.remove(x2_out)
                    reachable = compute_reachable(s)
                    
                    # Try to add three elements
                    added = []
                    for x1 in range(1, 101):
                        if x1 not in s and x1 not in (x1_out, x2_out):
                            if (200 - x1) not in reachable:
                                s.add(x1)
                                new_sums = {y + x1 for y in reachable if y + x1 <= 200}
                                reachable |= new_sums
                                added.append(x1)
                                break
                    
                    if len(added) >= 1:
                        for x2 in range(1, 101):
                            if x2 not in s and x2 not in (x1_out, x2_out) and x2 != added[0]:
                                if (200 - x2) not in reachable:
                                    s.add(x2)
                                    new_sums = {y + x2 for y in reachable if y + x2 <= 200}
                                    reachable |= new_sums
                                    added.append(x2)
                                    break
                    
                    if len(added) >= 2:
                        for x3 in range(1, 101):
                            if x3 not in s and x3 not in (x1_out, x2_out) and x3 not in added:
                                if (200 - x3) not in reachable:
                                    s.add(x3)
                                    new_sums = {y + x3 for y in reachable if y + x3 <= 200}
                                    reachable |= new_sums
                                    added.append(x3)
                                    break
                    
                    if len(added) == 3:
                        improved = True
                        break
                    else:
                        # Undo
                        for x_add in added:
                            s.discard(x_add)
                        reachable = compute_reachable(s)
                
                if improved:
                    break
            
            if not improved:
                # No improving move found, try random restarts
                if random.random() < 0.1:
                    # Random perturbation: remove a random element
                    if s:
                        x_remove = random.choice(list(s))
                        s.remove(x_remove)
                        reachable = compute_reachable(s)
        
        return s
    
    # Apply local search to best_set
    if time.time() < deadline:
        improved_set = try_local_search(best_set, max_iterations=500)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set
    
    # Try local search on some greedy results too
    if time.time() < deadline:
        for order in [list(range(100, 0, -1)), list(range(1, 101))]:
            if time.time() > deadline:
                break
            
            s = set()
            reachable = {0}
            for x in order:
                if (200 - x) not in reachable:
                    s.add(x)
                    new_sums = {y + x for y in reachable if y + x <= 200}
                    reachable |= new_sums
            
            if len(s) >= best_size - 2:  # Only try if close to best
                improved_set = try_local_search(s, max_iterations=200)
                if len(improved_set) > best_size and is_valid(improved_set):
                    best_size = len(improved_set)
                    best_set = improved_set
    
    # Final validation
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
