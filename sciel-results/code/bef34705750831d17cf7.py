import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 60  # Safety margin
    
    N = 256
    # Initial greedy construction
    S = set()
    rejected = []
    
    # Precompute nothing, just iterate
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        can_add = True
        # Check against all pairs in S
        # Optimization: if S is empty or size 1, no pairs, so always can add
        if len(S) >= 2:
            # Get list of elements
            elems = list(S)
            for i in range(len(elems)):
                for j in range(i, len(elems)):
                    y1 = elems[i]
                    y2 = elems[j]
                    if (y1 + y2) % x == 0:
                        can_add = False
                        break
                if not can_add:
                    break
        if can_add:
            S.add(x)
        else:
            rejected.append(x)
            
    # Local search: try to replace one element with two
    # We will iterate a few times
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle rejected to try different orders? 
        # Or just iterate through S
        # We need to be careful with the order of checking
        for s in list(S):
            if time.time() > deadline - 5:
                break
            
            S_temp = S - {s}
            # Try to add elements from rejected
            # We want to add as many as possible, preferably large ones?
            # Let's try to add elements from rejected in descending order
            added_count = 0
            new_elements = []
            for r in sorted(rejected, reverse=True):
                if time.time() > deadline - 5:
                    break
                # Check if r can be added to S_temp
                can_add_r = True
                if len(S_temp) >= 2:
                    elems_temp = list(S_temp)
                    for i in range(len(elems_temp)):
                        for j in range(i, len(elems_temp)):
                            if (elems_temp[i] + elems_temp[j]) % r == 0:
                                can_add_r = False
                                break
                        if not can_add_r:
                            break
                if can_add_r:
                    S_temp.add(r)
                    added_count += 1
                    new_elements.append(r)
                    # We only need to add 2 to improve size (remove 1, add 2 => net +1)
                    # But we should check if adding a 3rd is possible?
                    # Let's just see if we can add at least 2.
                    # Actually, we should try to add as many as possible to maximize gain.
                    # But adding more makes it harder to check.
                    # Let's stop after adding 2 for simplicity of the swap logic?
                    # No, if we can add 3, that's better.
                    # But the check is O(|S|^2).
                    # Let's just greedily add as many as possible from the sorted rejected list.
            
            if added_count >= 2:
                # This replacement is beneficial
                S = S_temp
                # Update rejected list
                # Remove new_elements from rejected
                for ne in new_elements:
                    if ne in rejected:
                        rejected.remove(ne)
                # Add s to rejected
                if s not in rejected:
                    rejected.append(s)
                improved = True
                # Restart the loop or continue?
                # Continue is fine, but the set changed.
                # Breaking to restart ensures we check new state properly.
                break
                
    return list(S)
