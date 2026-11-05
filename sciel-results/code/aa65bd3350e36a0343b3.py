import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # ap_list: list of tuples (a, b, c, d) with 1-based indexing
    # ap_index: dict mapping (a,b,c,d) to index? Or just list of lists of indices.
    # We need for each element x, the list of AP indices it belongs to.
    
    ap_list = []
    elem_to_aps = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(ap_idx)
            elem_to_aps[b].append(ap_idx)
            elem_to_aps[c].append(ap_idx)
            elem_to_aps[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # count_in_ap[i] = number of elements of AP i currently in the set
    count_in_ap = [0] * num_aps
    
    def add_element(x, in_set):
        """
        Adds x to the set if valid.
        Returns True if added, False if invalid.
        Modifies in_set and count_in_ap.
        """
        # Check validity
        for ap_idx in elem_to_aps[x]:
            if count_in_ap[ap_idx] == 3:
                # Adding x would make it 4
                return False
        
        # Valid, so add
        in_set.add(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] += 1
        return True

    def remove_element(x, in_set):
        """
        Removes x from the set.
        Modifies in_set and count_in_ap.
        """
        in_set.discard(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] -= 1

    best_size = 0
    best_set = set()
    
    # Run multiple restarts
    # Each restart:
    # 1. Greedy construction with random order
    # 2. Local search to improve
    
    for restart in range(200): # Max restarts, limited by time
        if time.time() > deadline:
            break
            
        current_set = set()
        
        # Greedy Phase
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            add_element(x, current_set)
            
        # Local Search Phase
        # We try to improve the set.
        # 1. Try to add any element not in set.
        # 2. If no additions, try swaps.
        
        improved = True
        local_iter = 0
        max_local_iters = 5000 # Limit iterations per restart to avoid infinite loops on single restart
        
        while improved and local_iter < max_local_iters and (time.time() - start_time) < deadline:
            improved = False
            local_iter += 1
            
            # 1. Try additions
            # We can just iterate through all not-in-set elements?
            # Or maintain a list of candidates?
            # For N=512, iterating through all 512 is fast enough.
            
            added_any = False
            for x in range(1, n + 1):
                if x not in current_set:
                    if add_element(x, current_set):
                        improved = True
                        added_any = True
                        # We can break here if we want to restart the addition scan
                        # or continue to add more in this pass
                        # Continuing is better as it might allow more additions
            if added_any:
                continue # Restart the loop to check for more additions or swaps
                
            # 2. No additions possible. Try swaps.
            # We want to find a swap (remove y, add x) that makes the set "better".
            # Heuristic: Minimize the number of APs with count 3.
            # Current "badness" = sum(1 for c in count_in_ap if c == 3)
            
            # To speed up swap search, we only consider swaps that break a blocking AP.
            # A blocking AP is one with count 3.
            # If we remove y from a blocking AP, that AP is no longer blocking.
            # Then we try to add x.
            
            # Find all blocking APs
            blocking_aps = [i for i, c in enumerate(count_in_ap) if c == 3]
            
            if not blocking_aps:
                # No blocking APs? Then we should have been able to add something?
                # Unless all elements are in the set?
                # If we can't add anything and there are no blocking APs, 
                # it means for every x not in set, it forms a 4-AP with 3 elements in set?
                # Wait, if count_in_ap[ap] < 3 for all ap containing x, then x can be added.
                # So if we can't add x, there must be at least one ap containing x with count 3.
                # So blocking_aps should not be empty if we are stuck.
                # Unless the set is full?
                if len(current_set) == n:
                    break
                # Otherwise, we are in a weird state, break to avoid infinite loop
                break

            # For each blocking AP, we can try removing one of its 3 elements in the set.
            # Then try to add an element x that was previously blocked by this AP (or another).
            
            # This is getting complex. Simpler swap strategy:
            # Randomly pick a blocking AP.
            # Pick one of its 3 elements y in current_set to remove.
            # Remove y.
            # Try to add any x not in set (that might have been blocked by this AP).
            # If we can add x, accept the swap.
            # If not, revert.
            
            # To make it more directed:
            # The elements x that were blocked by this specific AP are those x such that
            # x is in the AP? No, x is not in the AP.
            # x is blocked if x completes *some* AP.
            # If we remove y from AP A, AP A is no longer count 3.
            # So any x that was *only* blocked by AP A (and no other count-3 AP) can now be added.
            
            # Let's just try:
            # 1. Pick random blocking AP index `b_idx`.
            # 2. Get the 3 elements in `current_set` for this AP.
            # 3. Pick one `y` to remove.
            # 4. Remove `y`.
            # 5. Try to add any `x` not in set.
            # 6. If added, accept.
            # 7. If not, revert (re-add y).
            
            b_idx = random.choice(blocking_aps)
            a, b, c, e = ap_list[b_idx]
            # Find which of a,b,c,e are in current_set
            in_ap = [v for v in (a, b, c, e) if v in current_set]
            
            if len(in_ap) != 3:
                # Should not happen
                continue
                
            y = random.choice(in_ap)
            
            # Remove y
            remove_element(y, current_set)
            
            # Try to add any x
            added_x = None
            # Prioritize x that were involved in the blocking AP?
            # Actually, just try all x not in set.
            # To speed up, we can check x in random order.
            candidates_to_add = [x for x in range(1, n + 1) if x not in current_set]
            random.shuffle(candidates_to_add)
            
            for x in candidates_to_add:
                if add_element(x, current_set):
                    added_x = x
                    break
            
            if added_x is not None:
                # Swap successful (size unchanged, but configuration changed)
                improved = True
                # We accept this swap.
                # Note: We didn't increase size, but we might enable future additions.
                # To prevent oscillation, we could track history, but with randomization it's usually fine.
            else:
                # Swap failed, revert
                add_element(y, current_set)
                # Note: add_element checks validity. Since we just removed y and didn't add anything,
                # the state is back to before removal, so adding y should be valid.
                
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return sorted(best_set)
