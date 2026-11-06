import random
import time
import numpy as np

def construct():
    N = 128
    # Precompute all 4-term APs
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            ap = (a - 1, b - 1, c - 1, e - 1) # 0-indexed
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                index_to_aps[idx].append(ap_idx)
                
    num_aps = len(aps)
    ap_indices = np.array(aps, dtype=np.int32) # shape (num_aps, 4)
    
    # Helper to check if a 4-tuple is bad (has < 3 distinct colors)
    def is_bad_tuple(c0, c1, c2, c3):
        # Fast check
        if c0 == c1 == c2 == c3:
            return True
        if c0 == c1 == c2 and c3 != c0:
            return True
        if c0 == c1 == c3 and c2 != c0:
            return True
        if c0 == c2 == c3 and c1 != c0:
            return True
        if c1 == c2 == c3 and c0 != c1:
            return True
        if c0 == c1 and c2 == c3 and c0 != c2:
            return True
        if c0 == c2 and c1 == c3 and c0 != c1:
            return True
        if c0 == c3 and c1 == c2 and c0 != c1:
            return True
        return False

    # Precompute bad lookup for speed? 
    # For K=4, K^4 = 256. For K=5, 625.
    # We can use a set of bad tuples or a boolean array if we encode.
    # Let's stick to the function for clarity, or optimize with a set.
    # Since we call it often, a set of encoded keys is faster.
    
    def try_min_conflicts(K, time_limit=10.0):
        start_time = time.time()
        
        # Precompute bad keys
        bad_keys = set()
        for c0 in range(K):
            for c1 in range(K):
                for c2 in range(K):
                    for c3 in range(K):
                        if is_bad_tuple(c0, c1, c2, c3):
                            key = ((c0 * K + c1) * K + c2) * K + c3
                            bad_keys.add(key)
        
        max_restarts = 20
        best_solution = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize random coloring
            colors = np.random.randint(0, K, N, dtype=np.int32)
            
            # Calculate initial conflicts
            # conflicts[i] = number of bad APs involving index i
            conflicts = np.zeros(N, dtype=np.int32)
            # ap_bad[ap_idx] = 1 if bad, 0 if good
            ap_bad = np.zeros(num_aps, dtype=np.int32)
            
            current_conflicts = 0
            for ap_idx in range(num_aps):
                c0, c1, c2, c3 = colors[ap_indices[ap_idx, 0]], colors[ap_indices[ap_idx, 1]], colors[ap_indices[ap_idx, 2]], colors[ap_indices[ap_idx, 3]]
                key = ((c0 * K + c1) * K + c2) * K + c3
                if key in bad_keys:
                    ap_bad[ap_idx] = 1
                    current_conflicts += 1
                    for idx in ap_indices[ap_idx]:
                        conflicts[idx] += 1
            
            if current_conflicts == 0:
                return colors.tolist()
            
            # MinConflicts Loop
            max_steps = 100000 # Cap steps to avoid infinite loop in one restart
            steps = 0
            
            while current_conflicts > 0 and steps < max_steps:
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random index involved in a conflict
                # To be efficient, we can pick randomly from 0..N-1 and check if conflicts[i] > 0
                # Or maintain a list of conflicted indices.
                # For N=128, random pick is fine.
                i = np.random.randint(0, N)
                if conflicts[i] == 0:
                    continue
                
                # Try all K colors for i, pick the one with min new conflicts
                old_color = colors[i]
                min_conf = current_conflicts + 1000 # Init high
                best_new_color = old_color
                
                # Calculate conflicts for each new color
                # We need to evaluate the change in conflicts for APs involving i
                # Let's just calculate total conflicts for the affected APs for each candidate color
                
                # Get APs involving i
                ap_idxs = index_to_aps[i]
                
                for new_color in range(K):
                    if new_color == old_color:
                        continue
                    
                    # Calculate delta conflicts
                    # We need to check each AP involving i
                    # For each AP, determine if it was bad before and if it is bad now
                    # "Before" state is known from ap_bad array? 
                    # No, ap_bad is global. We need to simulate the change.
                    
                    # Let's store the state of APs involving i
                    # We can just re-evaluate the APs involving i with the new color
                    
                    delta = 0
                    # Save old badness of APs involving i
                    old_bad_states = []
                    for ap_idx in ap_idxs:
                        old_bad_states.append(ap_bad[ap_idx])
                        
                    # Apply new color temporarily
                    colors[i] = new_color
                    
                    # Update badness for APs involving i
                    for ap_idx in ap_idxs:
                        c0, c1, c2, c3 = colors[ap_indices[ap_idx, 0]], colors[ap_indices[ap_idx, 1]], colors[ap_indices[ap_idx, 2]], colors[ap_indices[ap_idx, 3]]
                        key = ((c0 * K + c1) * K + c2) * K + c3
                        is_bad = 1 if key in bad_keys else 0
                        
                        # Update global current_conflicts and conflicts array?
                        # It's easier to just calculate the total conflicts for these APs and compare to old
                        # But we need the "total" conflicts to decide.
                        # Let's calculate the number of bad APs among ap_idxs with new color
                        # And compare to number of bad APs among ap_idxs with old color
                        pass
                    
                    # This inner loop is O(degree(i) * K). Degree is ~10-20. K=4-5.
                    # So ~100 ops per step. 100k steps -> 10M ops. Fast.
                    
                    # Let's refine the delta calculation
                    # We need to know how many of the APs in ap_idxs are bad with old_color
                    # and how many are bad with new_color.
                    # The rest of the APs (not involving i) don't change.
                    # So total_conflicts_new = total_conflicts_old - bad_old_in_ap_idxs + bad_new_in_ap_idxs
                    
                    bad_old_count = sum(old_bad_states)
                    bad_new_count = 0
                    # Re-evaluate APs with new_color (which is currently in colors[i])
                    for ap_idx in ap_idxs:
                        c0, c1, c2, c3 = colors[ap_indices[ap_idx, 0]], colors[ap_indices[ap_idx, 1]], colors[ap_indices[ap_idx, 2]], colors[ap_indices[ap_idx, 3]]
                        key = ((c0 * K + c1) * K + c2) * K + c3
                        if key in bad_keys:
                            bad_new_count += 1
                    
                    new_total = current_conflicts - bad_old_count + bad_new_count
                    if new_total < min_conf:
                        min_conf = new_total
                        best_new_color = new_color
                    
                    # Revert colors[i] to old_color for next candidate
                    colors[i] = old_color
                
                # Apply best_new_color
                if best_new_color != old_color:
                    # Update global state
                    # Calculate bad_old_count again? We have it from the loop? 
                    # We didn't store it. Let's re-calc or store.
                    # For simplicity, let's just re-calc the delta properly in the acceptance step.
                    
                    # Get old bad states
                    old_bad_states = []
                    for ap_idx in index_to_aps[i]:
                        old_bad_states.append(ap_bad[ap_idx])
                    bad_old_count = sum(old_bad_states)
                    
                    # Apply new color
                    colors[i] = best_new_color
                    
                    # Update bad states for APs involving i
                    bad_new_count = 0
                    for ap_idx in index_to_aps[i]:
                        c0, c1, c2, c3 = colors[ap_indices[ap_idx, 0]], colors[ap_indices[ap_idx, 1]], colors[ap_indices[ap_idx, 2]], colors[ap_indices[ap_idx, 3]]
                        key = ((c0 * K + c1) * K + c2) * K + c3
                        is_bad = 1 if key in bad_keys else 0
                        
                        # Update ap_bad and conflicts
                        if ap_bad[ap_idx] != is_bad:
                            if is_bad:
                                ap_bad[ap_idx] = 1
                                current_conflicts += 1
                                for idx in ap_indices[ap_idx]:
                                    conflicts[idx] += 1
                            else:
                                ap_bad[ap_idx] = 0
                                current_conflicts -= 1
                                for idx in ap_indices[ap_idx]:
                                    conflicts[idx] -= 1
                        if is_bad:
                            bad_new_count += 1
                            
                    # Wait, the update logic above is messy.
                    # Let's simplify:
                    # 1. Store old ap_bad for APs involving i.
                    # 2. Set colors[i] = best_new_color.
                    # 3. For each AP in index_to_aps[i]:
                    #      Calculate new is_bad.
                    #      If new is_bad != old ap_bad[ap_idx]:
                    #          Update current_conflicts and conflicts[] for all 4 indices in AP.
                    #          Set ap_bad[ap_idx] = new is_bad.
                    
                    # Let's redo the apply step cleanly
                    # We already reverted colors[i] to old_color in the candidate loop.
                    # So we are back to old state.
                    
                    # 1. Get old bad states
                    old_bad_vals = [ap_bad[ap_idx] for ap_idx in index_to_aps[i]]
                    
                    # 2. Apply new color
                    colors[i] = best_new_color
                    
                    # 3. Update
                    for ap_idx, old_bad in zip(index_to_aps[i], old_bad_vals):
                        c0, c1, c2, c3 = colors[ap_indices[ap_idx, 0]], colors[ap_indices[ap_idx, 1]], colors[ap_indices[ap_idx, 2]], colors[ap_indices[ap_idx, 3]]
                        key = ((c0 * K + c1) * K + c2) * K + c3
                        new_bad = 1 if key in bad_keys else 0
                        
                        if new_bad != old_bad:
                            if new_bad:
                                ap_bad[ap_idx] = 1
                                current_conflicts += 1
                                for idx in ap_indices[ap_idx]:
                                    conflicts[idx] += 1
                            else:
                                ap_bad[ap_idx] = 0
                                current_conflicts -= 1
                                for idx in ap_indices[ap_idx]:
                                    conflicts[idx] -= 1
                            
                steps += 1
                
                if current_conflicts == 0:
                    return colors.tolist()
                    
        return None

    # Try increasing number of colors
    for K in [4, 5, 6]:
        sol = try_min_conflicts(K, time_limit=15.0)
        if sol is not None:
            return sol
            
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
