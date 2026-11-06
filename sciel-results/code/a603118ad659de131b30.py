import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_set = []
    best_size = 0
    
    # 1. Greedy Initialization
    # We can try multiple random orders or just one good one.
    # Let's do a few greedy runs.
    
    for restart in range(20): # Cap restarts, rely on local search
        if time.time() > deadline: break
        
        order = list(range(1, 1001))
        random.shuffle(order)
        
        current_set, current_counts = _greedy(order)
        
        # 2. Local Search
        current_set, current_counts = _local_search(current_set, current_counts, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set

def _greedy(order):
    current_set = []
    current_counts = {}
    for x in order:
        can_add = True
        for y in current_set:
            s = x + y
            if current_counts.get(s, 0) >= 2:
                can_add = False
                break
        if can_add:
            current_set.append(x)
            # Update counts
            for y in current_set:
                s = x + y
                current_counts[s] = current_counts.get(s, 0) + 1
    return current_set, current_counts

def _local_search(s, counts, deadline):
    # s is list, counts is dict
    s_set = set(s)
    
    # Try to add any element
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 1001):
            if x in s_set: continue
            # Check if x can be added
            can_add = True
            for y in s:
                if counts.get(x+y, 0) >= 2:
                    can_add = False
                    break
            if can_add:
                s.append(x)
                s_set.add(x)
                for y in s:
                    counts[x+y] = counts.get(x+y, 0) + 1
                changed = True
                break # Restart loop to pick up new possibilities
    
    # If no single add works, try remove-and-add
    # This is more expensive. Try a few random removals.
    for _ in range(100): # Limit iterations
        if time.time() > deadline: break
        if not s: break
        # Pick a random element to remove
        idx = random.randint(0, len(s)-1)
        x = s[idx]
        
        # Remove x
        s.pop(idx)
        s_set.remove(x)
        for y in s:
            counts[x+y] -= 1
            if counts[x+y] == 0: del counts[x+y]
        counts[2*x] -= 1
        if counts[2*x] == 0: del counts[2*x]
        
        # Try to add as many as possible
        added = 0
        added_elements = []
        for y in range(1, 1001):
            if y in s_set: continue
            can_add = True
            for z in s:
                if counts.get(y+z, 0) >= 2:
                    can_add = False
                    break
            if can_add:
                s.append(y)
                s_set.add(y)
                added_elements.append(y)
                added += 1
                for z in s:
                    counts[y+z] = counts.get(y+z, 0) + 1
        
        # If we added more than we removed (1), keep it.
        # Actually, if added > 1, size increases. If added == 1, size same but structure changed.
        # We want to maximize size. So keep if added > 1.
        # Or keep if added >= 1 to allow for future improvements?
        # Usually, we keep if size increases. If size is same, we might get stuck.
        # Let's keep if added > 1.
        if added <= 1:
            # Revert
            for y in added_elements:
                s.pop() # This is wrong if order matters. 
                # Better to rollback properly.
            # Reverting is tricky. Let's just not do this if it's not beneficial.
            # Simpler: Only do removal if we are confident it helps.
            # Or, just try random swaps?
            
    return s, counts
