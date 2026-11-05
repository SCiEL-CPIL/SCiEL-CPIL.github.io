import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0 # 90s limit, leave 5s margin
    
    # 1. Greedy Initialization
    # Try a few different greedy strategies and keep the best
    def greedy_init(order):
        selected = []
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not (new_reach >> 200) & 1:
                selected.append(x)
                reachable = new_reach
        return set(selected)
    
    # Strategies: Descending, Ascending, Random Shuffles
    best_init = set()
    best_init_size = 0
    
    # Descending
    s1 = greedy_init(range(100, 0, -1))
    if len(s1) > best_init_size:
        best_init = s1
        best_init_size = len(s1)
        
    # Ascending
    s2 = greedy_init(range(1, 101))
    if len(s2) > best_init_size:
        best_init = s2
        best_init_size = len(s2)
        
    # Random shuffles
    for _ in range(10):
        lst = list(range(1, 101))
        random.shuffle(lst)
        s3 = greedy_init(lst)
        if len(s3) > best_init_size:
            best_init = s3
            best_init_size = len(s3)
            
    current_set = best_init.copy()
    current_reachable = 1
    for x in current_set:
        current_reachable |= (current_reachable << x)
        
    best_set = current_set.copy()
    best_size = len(current_set)
    
    all_elements = list(range(1, 101))
    
    def recompute_reachable(s):
        r = 1
        for x in s:
            r |= (r << x)
            if (r >> 200) & 1:
                return None # Invalid
        return r

    while time.time() < deadline:
        # Phase 1: Greedy Add
        improved = True
        while improved:
            improved = False
            # Shuffle to break symmetry in addition order? 
            # Or just iterate.
            for x in all_elements:
                if x not in current_set:
                    new_reach = current_reachable | (current_reachable << x)
                    if not (new_reach >> 200) & 1:
                        current_set.add(x)
                        current_reachable = new_reach
                        improved = True
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(current_set)
                        break # Restart scan as set changed
        
        # Phase 2: Swap Search
        # Find a valid swap (remove u, add v)
        # To speed up, we can sample swaps rather than checking all if stuck
        # But first, let's check if any swap improves the "potential"
        # Since size is constant, we just want to move to a different local optimum.
        
        # Check for valid swaps
        # We pick a random u from current_set and random v from not in current_set
        # If we find a valid swap, we accept it with some probability (or always if we want to explore)
        # To avoid infinite loops on plateaus, we can keep a history or just randomize.
        
        # Let's try to find a swap that allows future growth.
        # Heuristic: Replace u with v if v > u? Or just random.
        
        found_swap = False
        # Try a limited number of random swaps to find a valid one
        # If we find one, apply it.
        for _ in range(100): # Limit checks per iteration to avoid stalling
            if time.time() > deadline:
                break
            if not current_set:
                break
            u = random.choice(list(current_set))
            candidates = [x for x in all_elements if x not in current_set]
            if not candidates:
                break
            v = random.choice(candidates)
            
            # Check validity of S - {u} + {v}
            new_set = current_set.copy()
            new_set.remove(u)
            new_set.add(v)
            
            # Recompute reachable
            new_reach = 1
            valid = True
            # Optimization: if u was the only thing preventing 200? No, bitsets don't work that way.
            for x in new_set:
                new_reach |= (new_reach << x)
                if (new_reach >> 200) & 1:
                    valid = False
                    break
            
            if valid:
                # Accept the swap to move to a new position
                current_set = new_set
                current_reachable = new_reach
                found_swap = True
                break
        
        if not found_swap:
            # Stuck in local optimum. 
            # Try a "destructive" move: remove a random element to allow new additions.
            # This reduces size temporarily, but might lead to a larger size later.
            if current_set:
                u = random.choice(list(current_set))
                current_set.remove(u)
                # Recompute reachable for the smaller set
                current_reachable = 1
                for x in current_set:
                    current_reachable |= (current_reachable << x)
                # The next loop iteration will run Phase 1 (Greedy Add) again
            else:
                # Empty set, add a random element
                current_set.add(random.choice(all_elements))
                current_reachable = 1 | (1 << list(current_set)[0])

    return list(best_set)
