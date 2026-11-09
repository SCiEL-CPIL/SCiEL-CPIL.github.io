import time
import random

def construct():
    N = 1000
    MAX_SUM = 2 * N
    MAX_REPS = 2
    
    # 1. Greedy Initialization
    S = set()
    sum_counts = [0] * (MAX_SUM + 1)
    
    for x in range(1, N + 1):
        valid = True
        if sum_counts[2*x] >= MAX_REPS:
            valid = False
        else:
            for y in S:
                s = x + y
                if sum_counts[s] >= MAX_REPS:
                    valid = False
                    break
        
        if valid:
            S.add(x)
            sum_counts[2*x] += 1
            for y in S:
                if y == x: continue
                sum_counts[x+y] += 1
    
    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    timeout = 140.0
    
    # 2. Local Search
    # We alternate between trying to Add and trying to Swap.
    # Add increases size. Swap maintains size but changes structure.
    
    # To speed up "try add", we can pre-filter candidates?
    # Not easily, as counts change.
    
    # We will loop until no improvement (no add) and no successful swap that we think is "better"?
    # Since swap doesn't change size, how do we know it's "better"?
    # We don't. So we accept any valid swap to explore the space.
    # To prevent infinite loops of swaps, we can limit the number of swaps per "add phase".
    # Or just rely on the time limit.
    
    # Strategy:
    # 1. Try to Add. If successful, update best_S, reset swap counter.
    # 2. If no Add, try to Swap. If successful, increment swap counter.
    # 3. If swap counter > limit, maybe restart or change strategy?
    #    For now, just keep swapping until time runs out or no swaps possible.
    
    no_add_streak = 0
    max_swaps_per_add = 100  # Heuristic: try 100 swaps before giving up on this local minimum?
    # Actually, if we can't add, and we swap, we might enable an add.
    # So we should keep swapping.
    
    while (time.time() - start_time) < timeout:
        # Try Add
        added = False
        # Shuffle candidates to avoid bias
        candidates_add = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates_add)
        
        for x in candidates_add:
            if (time.time() - start_time) > timeout:
                break
                
            if sum_counts[2*x] >= MAX_REPS:
                continue
                
            valid = True
            for y in S:
                s = x + y
                if sum_counts[s] >= MAX_REPS:
                    valid = False
                    break
            
            if valid:
                # Add x
                S.add(x)
                sum_counts[2*x] += 1
                for y in S:
                    if y == x: continue
                    sum_counts[x+y] += 1
                
                best_S = set(S)
                best_size = len(S)
                added = True
                no_add_streak = 0
                break
        
        if added:
            continue
        
        # Try Swap
        # Pick a random x in S to remove
        # Then try to add any y not in S
        # If found, accept.
        
        list_S = list(S)
        random.shuffle(list_S)
        
        swapped = False
        for x in list_S:
            if (time.time() - start_time) > timeout:
                break
                
            # Remove x
            S.remove(x)
            sum_counts[2*x] -= 1
            for y in S:
                sum_counts[x+y] -= 1
            
            # Try to add y
            candidates_add = [y for y in range(1, N + 1) if y not in S]
            random.shuffle(candidates_add)
            
            found_add = False
            for y in candidates_add:
                if (time.time() - start_time) > timeout:
                    break
                    
                if sum_counts[2*y] >= MAX_REPS:
                    continue
                    
                valid = True
                for z in S:
                    s = y + z
                    if sum_counts[s] >= MAX_REPS:
                        valid = False
                        break
                
                if valid:
                    # Accept swap
                    S.add(y)
                    sum_counts[2*y] += 1
                    for z in S:
                        if z == y: continue
                        sum_counts[y+z] += 1
                    found_add = True
                    swapped = True
                    break
            
            if not found_add:
                # Revert
                S.add(x)
                sum_counts[2*x] += 1
                for y in S:
                    if y == x: continue
                    sum_counts[x+y] += 1
            
            if swapped:
                # We made a swap. Continue the outer loop to try Add again.
                break
                
        if not swapped:
            # No adds, no swaps possible. Stuck.
            # Could try random restarts or perturbation, but for now, stop.
            break

    return sorted(best_S)
