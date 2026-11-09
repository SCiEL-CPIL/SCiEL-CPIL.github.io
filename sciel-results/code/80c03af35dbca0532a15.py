import random
import time

def construct():
    n = 128
    
    # Precompute all 4-APs
    aps = []
    ap_to_pos = [[] for _ in range(n)] # ap_to_pos[i] = list of AP indices containing i
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            ap_idx = len(aps)
            aps.append((i1, i2, i3, i4))
            ap_to_pos[i1].append(ap_idx)
            ap_to_pos[i2].append(ap_idx)
            ap_to_pos[i3].append(ap_idx)
            ap_to_pos[i4].append(ap_idx)
    
    num_aps = len(aps)
    
    def is_valid_ap(ap_idx, colors):
        i1, i2, i3, i4 = aps[ap_idx]
        c1 = colors[i1]
        c2 = colors[i2]
        c3 = colors[i3]
        c4 = colors[i4]
        return len({c1, c2, c3, c4}) >= 3
    
    def count_violations(colors):
        v = 0
        for i in range(num_aps):
            if not is_valid_ap(i, colors):
                v += 1
        return v
        
    def get_violating_aps(colors):
        v_aps = []
        for i in range(num_aps):
            if not is_valid_ap(i, colors):
                v_aps.append(i)
        return v_aps

    def search(k, time_limit=55):
        start_time = time.time()
        
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Initial violation count
            current_violations = count_violations(colors)
            if current_violations == 0:
                return colors
            
            for i in range(200000): # Iterations per attempt
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random violating AP
                viol_aps = get_violating_aps(colors)
                if not viol_aps:
                    return colors
                
                ap_idx = random.choice(viol_aps)
                i1, i2, i3, i4 = aps[ap_idx]
                pos = random.choice([i1, i2, i3, i4])
                
                old_color = colors[pos]
                
                # Try changing pos to each color
                best_color = old_color
                best_delta = 0 # We want to minimize violations
                
                # We need to evaluate the change in violations.
                # Only APs containing 'pos' are affected.
                affected_aps = ap_to_pos[pos]
                
                # Current status of affected APs
                # Note: We only care about the net change.
                
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    new_violations = 0
                    for a_idx in affected_aps:
                        if not is_valid_ap(a_idx, colors):
                            new_violations += 1
                    # We don't have the old violation count for these specific APs easily stored?
                    # We can just count total violations? No, that's O(N^2).
                    # We can store a boolean array `ap_is_valid`?
                    
                    # Let's stick to a simpler heuristic: 
                    # Just count total violations if k is small and N is small?
                    # N=128, Num APs ~ 2600.
                    # Counting all is ~2600 ops. 200k iters * 2600 = 500M ops. Too slow in Python.
                    
                    # Optimization: Maintain a list of violating APs?
                    # Or just count the delta on affected APs.
                    
                    # Let's revert and try a different approach: 
                    # Store `ap_valid` array.
                    # Update it incrementally.
                    
                    colors[pos] = old_color # Revert for next trial
                    break # This loop logic is getting messy. Let's restart the search function design.
            
            # If we exit the inner loop without finding 0, we can try to continue with the best state found?
            # Or just restart.
            
        return None

    # Better Search Implementation
    def search_v2(k, time_limit=55):
        start_time = time.time()
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(n)]
            ap_valid = [is_valid_ap(i, colors) for i in range(num_aps)]
            current_violations = num_aps - sum(ap_valid)
            
            if current_violations == 0:
                return colors
            
            for i in range(1000000):
                if time.time() - start_time > time_limit:
                    break
                
                if current_violations == 0:
                    return colors
                
                # Pick a random violating AP
                # To avoid O(N) scan for a random violating AP, we can maintain a list?
                # Or just pick a random AP and if it's valid, pick a random position?
                # Strategy: Pick random position, if it's in a violation, good. Else random.
                
                # Let's pick a random position
                pos = random.randint(0, n - 1)
                
                # Check if this position is in any violating AP
                # If not, changing it might create violations or not.
                # If yes, changing it might fix some.
                
                old_color = colors[pos]
                
                best_color = old_color
                best_violations = current_violations
                
                # Evaluate each color
                for c in range(k):
                    if c == old_color:
                        continue
                    
                    colors[pos] = c
                    
                    # Update ap_valid for affected APs
                    delta = 0
                    for a_idx in ap_to_pos[pos]:
                        old_status = ap_valid[a_idx]
                        new_status = is_valid_ap(a_idx, colors)
                        if old_status and not new_status:
                            delta += 1
                        elif not old_status and new_status:
                            delta -= 1
                        # Update ap_valid
                        ap_valid[a_idx] = new_status
                    
                    new_violations = current_violations + delta
                    
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_color = c
                        # We can't keep the state change easily if we try multiple colors
                        # We need to revert ap_valid if we don't pick this color
                        # This is getting complicated to manage state rollback.
                        
                    # Revert ap_valid changes for this trial
                    # We need to store the old states
                    # Let's just do a full recount if k is small?
                    # No, let's just pick the first color that reduces violations?
                    # Or just pick a random color if no improvement?
                    
                    # Simple WalkSAT: 
                    # If we found an improvement, commit it.
                    # If not, pick a random color.
                    
                    # To make it simpler:
                    # 1. Pick pos.
                    # 2. Try all colors, find the one with min violations (using delta).
                    # 3. If min < current, commit.
                    # 4. Else, commit a random color.
                    
                    # We need to restore ap_valid after the loop if we didn't commit?
                    # Or we can just compute the best color, then apply it and update ap_valid once.
                    
                    colors[pos] = old_color # Revert color
                    # We need to restore ap_valid to its state before this 'for c' loop
                    # This requires storing the old states of affected APs.
                    
                    # Let's restructure:
                    # Store old states
                    old_states = []
                    for a_idx in ap_to_pos[pos]:
                        old_states.append(ap_valid[a_idx])
                        
                    # Now try colors
                    best_c = old_color
                    best_v = current_violations
                    for c in range(k):
                        if c == old_color: continue
                        colors[pos] = c
                        delta = 0
                        for j, a_idx in enumerate(ap_to_pos[pos]):
                            new_status = is_valid_ap(a_idx, colors)
                            if old_states[j] and not new_status: delta += 1
                            elif not old_states[j] and new_status: delta -= 1
                        v = current_violations + delta
                        if v < best_v:
                            best_v = v
                            best_c = c
                        # Don't update ap_valid yet
                    
                    # Apply best_c
                    colors[pos] = best_c
                    for j, a_idx in enumerate(ap_to_pos[pos]):
                        new_status = is_valid_ap(a_idx, colors)
                        if old_states[j] and not new_status: current_violations += 1
                        elif not old_states[j] and new_status: current_violations -= 1
                        ap_valid[a_idx] = new_status
                        
                    # If no improvement was found (best_c == old_color), we should still make a random move to escape local minima?
                    # In the loop above, if no c < current_violations, best_c remains old_color.
                    # If best_c == old_color, we didn't change anything.
                    # We should force a random change if stuck.
                    if best_c == old_color:
                        # Pick a random color different from old
                        new_c = random.choice([c for c in range(k) if c != old_color])
                        # Apply it
                        colors[pos] = new_c
                        for j, a_idx in enumerate(ap_to_pos[pos]):
                            new_status = is_valid_ap(a_idx, colors)
                            if old_states[j] and not new_status: current_violations += 1
                            elif not old_states[j] and new_status: current_violations -= 1
                            ap_valid[a_idx] = new_status

        return None

    result = search_v2(3, time_limit=20)
    if result:
        return result

    result = search_v2(4, time_limit=20)
    if result:
        return result

    result = search_v2(5, time_limit=10)
    if result:
        return result

    # Fallback
    return [0]*n
