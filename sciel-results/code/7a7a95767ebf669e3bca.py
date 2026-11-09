import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # 1. Generate initial greedy solution
    # Greedy from 100 down to 1
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            selected.append(x)
            reachable |= new_sums
    
    best_set = set(selected)
    best_size = len(best_set)
    
    # Helper to check if a set is valid
    def is_valid(s):
        # Check if any subset sums to 200
        # This is expensive if done from scratch, but sets are small (<100)
        # We can use DP or bitset.
        # Since we only have 100 elements, we can just check subset sums.
        # But doing this repeatedly is slow.
        # We'll use the incremental reachable set logic if possible, 
        # but for arbitrary swaps, we might need to recompute.
        # Given N=100, recomputing reachable sums for a set of size ~50 is fast.
        r = {0}
        for x in s:
            new_r = {s_ + x for s_ in r if s_ + x <= 200}
            if 200 in new_r:
                return False
            r |= new_r
        return True

    # Helper to get reachable sums
    def get_reachable(s):
        r = {0}
        for x in s:
            new_r = {s_ + x for s_ in r if s_ + x <= 200}
            r |= new_r
        return r

    # 2. Local Search / Hill Climbing
    # Try to add elements from 1..50 not in best_set
    # Try to swap elements
    
    # Precompute reachable for current best
    # We'll maintain reachable incrementally if we just add/remove
    
    # Strategy: 
    # The greedy set is likely {51..100} minus some rejected.
    # Let's try to find a valid set of size > best_size.
    
    # We can try to add an element y from 1..50.
    # If we add y, we need 200-y not in reachable(best_set).
    
    current_reachable = get_reachable(best_set)
    
    # Try adding elements
    candidates_to_add = [x for x in range(1, 51) if x not in best_set]
    random.shuffle(candidates_to_add)
    
    for y in candidates_to_add:
        if time.time() > deadline:
            break
        if 200 - y in current_reachable:
            continue
        
        # Try adding y
        new_set = best_set | {y}
        # Check if valid (should be if 200-y not in reachable)
        # But wait, adding y might create new sums that block nothing, 
        # but we must ensure no OTHER subset sums to 200.
        # The condition "200 not in new_sums" where new_sums = {s+y} 
        # ensures that no sum involving y is 200.
        # Sums not involving y are already safe.
        # So if 200-y not in current_reachable, then adding y is safe.
        
        new_reachable = current_reachable | {s + y for s in current_reachable if s + y <= 200}
        if 200 not in new_reachable:
            best_set = new_set
            best_size += 1
            current_reachable = new_reachable
            # Restart add loop? Or just continue?
            # If we add one, we might be able to add another.
            # Let's break and restart the process to be safe and simple.
            # Or just continue scanning, but current_reachable is updated.
            # However, the list candidates_to_add is static.
            # Let's break and restart to allow for more complex interactions.
            break

    # Try swapping
    # If we can't add, maybe we can swap a large number for a small one?
    # Or swap to free up a "slot" for a better number?
    
    # Given the time, the simple "add if possible" might be sufficient to beat the greedy
    # if the greedy left some small numbers addable.
    
    # Let's do a few rounds of random restarts with perturbation
    # But we must keep the function fast.
    
    # The greedy solution is deterministic.
    # If the greedy solution is size 49, and we can add 1, we get 50.
    
    return list(best_set)
