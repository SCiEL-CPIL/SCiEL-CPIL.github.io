import time
import random

def construct():
        N = 128
        # Precompute APs
        aps = []
        for d in range(1, (N - 1) // 3 + 1):
            for i in range(N - 3 * d):
                aps.append((i, i + d, i + 2 * d, i + 3 * d))
        
        M = len(aps)
        
        # Map position to list of AP indices
        pos_to_aps = [[] for _ in range(N)]
        for idx, ap in enumerate(aps):
            for pos in ap:
                pos_to_aps[pos].append(idx)

        def solve_with_k(k, time_limit):
            start_time = time.time()
            # Random restarts
            for restart in range(100):
                if time.time() - start_time > time_limit:
                    return None
                
                # Initialize
                colors = [random.randint(0, k - 1) for _ in range(N)]
                
                # Calculate initial violations
                is_bad = [False] * M
                bad_count = 0
                for idx, ap in enumerate(aps):
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        is_bad[idx] = True
                        bad_count += 1
                
                # Local Search
                # Max steps per restart
                max_steps = 5000
                for step in range(max_steps):
                    if time.time() - start_time > time_limit:
                        return None
                    if bad_count == 0:
                        return colors
                    
                    # Pick a random bad AP
                    # Optimization: Keep a list of bad AP indices?
                    # Or just pick random AP and check if bad?
                    # If bad_count is small, picking random AP might miss it.
                    # Better: maintain a list/set of bad APs.
                    # But updating a set is slower.
                    # Given N=128, M is small (~2000).
                    # We can just pick a random bad AP from a precomputed list of currently bad ones?
                    # Let's just pick a random position involved in a bad AP.
                    
                    # Simple approach: Pick random pos.
                    # If it's in a bad AP, change it.
                    # To ensure progress, pick a pos from a random bad AP.
                    
                    # We need to efficiently pick a bad AP.
                    # Let's maintain a list of bad AP indices.
                    # Updating it is O(1) amortized if we swap-remove.
                    
                    # Let's try a simpler heuristic first:
                    # Pick random pos.
                    pos = random.randint(0, N - 1)
                    
                    # Check if pos is in any bad AP
                    # If not, move it randomly? Or skip?
                    # Better: Pick a random bad AP.
                    
                    # To pick a random bad AP efficiently without a dynamic list:
                    # We can just iterate? No.
                    # Let's maintain `bad_aps_list`.
                    
                    # Actually, let's just use the `is_bad` array and pick random APs until we find a bad one.
                    # If bad_count is high, this is fast.
                    # If bad_count is low, this is slow.
                    # When bad_count is low, we are close to solution.
                    
                    # Let's implement a "pick random bad AP" function.
                    # For simplicity and speed in Python, I'll just pick a random AP index and check `is_bad`.
                    # If not bad, retry a few times.
                    
                    found_bad_ap = False
                    for _ in range(10):
                        ap_idx = random.randint(0, M - 1)
                        if is_bad[ap_idx]:
                            ap = aps[ap_idx]
                            # Pick a random position in this AP
                            pos = random.choice(ap)
                            found_bad_ap = True
                            break
                    
                    if not found_bad_ap:
                        # If we couldn't find a bad AP in 10 tries, it's probably 0 or very low
                        # Just pick random pos
                        pos = random.randint(0, N - 1)

                    current_val = colors[pos]
                    
                    # Try to change to a better value
                    # Evaluate all k values
                    best_val = current_val
                    best_delta = 0 # Change in bad_count
                    
                    # We need to calculate the change in bad_count if we set colors[pos] to new_val
                    # This requires checking all APs containing pos.
                    
                    for new_val in range(k):
                        if new_val == current_val:
                            continue
                        
                        delta = 0
                        for ap_idx in pos_to_aps[pos]:
                            ap = aps[ap_idx]
                            # Get colors, replacing pos with new_val
                            c0 = colors[ap[0]] if ap[0] != pos else new_val
                            c1 = colors[ap[1]] if ap[1] != pos else new_val
                            c2 = colors[ap[2]] if ap[2] != pos else new_val
                            c3 = colors[ap[3]] if ap[3] != pos else new_val
                            
                            was_bad = is_bad[ap_idx]
                            is_bad_new = len({c0, c1, c2, c3}) < 3
                            
                            if was_bad and not is_bad_new:
                                delta -= 1
                            elif not was_bad and is_bad_new:
                                delta += 1
                            
                            # Early break if delta > best_delta?
                            # No, we need to check all APs to know total delta.
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_val = new_val
                    
                    # If no improvement, pick random
                    if best_delta >= 0:
                        # Simulated annealing / random kick
                        best_val = random.randint(0, k - 1)
                        if best_val == current_val:
                            best_val = (best_val + 1) % k
                        
                        # Recalculate delta for this random choice?
                        # Or just apply and update?
                        # Let's just apply and update counts directly to save time?
                        # No, we need to update is_bad correctly.
                        
                        # Let's just use the best_val found (which might be current if no improvement)
                        # But if no improvement, we MUST move to escape local minima.
                        # So pick a random different value.
                        
                        # Calculate delta for this random choice
                        delta = 0
                        for ap_idx in pos_to_aps[pos]:
                            ap = aps[ap_idx]
                            c0 = colors[ap[0]] if ap[0] != pos else best_val
                            c1 = colors[ap[1]] if ap[1] != pos else best_val
                            c2 = colors[ap[2]] if ap[2] != pos else best_val
                            c3 = colors[ap[3]] if ap[3] != pos else best_val
                            
                            was_bad = is_bad[ap_idx]
                            is_bad_new = len({c0, c1, c2, c3}) < 3
                            
                            if was_bad and not is_bad_new:
                                delta -= 1
                            elif not was_bad and is_bad_new:
                                delta += 1
                        best_delta = delta

                    # Apply change
                    colors[pos] = best_val
                    bad_count += best_delta
                    
                    # Update is_bad array
                    for ap_idx in pos_to_aps[pos]:
                        ap = aps[ap_idx]
                        c0 = colors[ap[0]]
                        c1 = colors[ap[1]]
                        c2 = colors[ap[2]]
                        c3 = colors[ap[3]]
                        is_bad[ap_idx] = len({c0, c1, c2, c3}) < 3

                # End steps
            # End restarts
            return None

        # Try k=3
        global_deadline = time.time() + 45.0
        time_per_k = 10.0
        
        for k in [3, 4, 5, 6]:
            if time.time() > global_deadline:
                break
            sol = solve_with_k(k, time_per_k)
            if sol is not None:
                return sol
        
        # Fallback
        return list(range(N))
