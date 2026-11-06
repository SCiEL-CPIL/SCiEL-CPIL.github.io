import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    # Precompute all non-constant 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0
    aps = []
    ap_indices_for_pos = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idxs = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            ap_id = len(aps)
            aps.append(idxs)
            for idx in idxs:
                ap_indices_for_pos[idx].append(ap_id)
                
    num_aps = len(aps)
    
    def count_violations(colors):
        """Returns the number of 4-APs that use at most 2 distinct colors."""
        count = 0
        for idxs in aps:
            c1, c2, c3, c4 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
            # Check if distinct colors < 3
            # This is equivalent to: not (at least 3 distinct)
            # At least 3 distinct means:
            # (c1 != c2 and c1 != c3 and c1 != c4) OR
            # (c2 != c3 and c2 != c4) OR
            # (c3 != c4) -- wait, simpler to just use set or logic
            # Let's use logic for speed
            # If all same: violation
            # If 3 same, 1 diff: violation (2 distinct)
            # If 2 same, 2 same: violation (2 distinct)
            # If 2 same, 1, 1: valid (3 distinct)
            # If 1, 1, 1, 1: valid (4 distinct)
            
            # Count distincts
            # c1, c2, c3, c4
            if c1 == c2 == c3 == c4:
                count += 1
            elif c1 == c2 == c3 and c3 != c4:
                count += 1
            elif c1 == c2 and c2 == c4 and c1 != c3:
                count += 1
            elif c1 == c3 and c3 == c4 and c1 != c2:
                count += 1
            elif c1 == c4 and c4 == c2 and c1 != c3:
                count += 1
            elif c1 == c2 and c3 == c4 and c1 != c3:
                count += 1
            elif c1 == c3 and c2 == c4 and c1 != c2:
                count += 1
            elif c1 == c4 and c2 == c3 and c1 != c2:
                count += 1
            else:
                # Valid: at least 3 distinct
                pass
        return count

    def is_ap_valid(colors, ap_id):
        idxs = aps[ap_id]
        c1, c2, c3, c4 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
        # Check if at least 3 distinct
        # Negation of "at most 2 distinct"
        if c1 == c2 == c3 == c4:
            return False
        if c1 == c2 == c3 and c3 != c4:
            return False
        if c1 == c2 and c2 == c4 and c1 != c3:
            return False
        if c1 == c3 and c3 == c4 and c1 != c2:
            return False
        if c1 == c4 and c4 == c2 and c1 != c3:
            return False
        if c1 == c2 and c3 == c4 and c1 != c3:
            return False
        if c1 == c3 and c2 == c4 and c1 != c2:
            return False
        if c1 == c4 and c2 == c3 and c1 != c2:
            return False
        return True

    best_colors = None
    best_k = 128
    
    # Try increasing number of colors k
    for k in range(3, 11):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this k
        found = False
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Count initial violations
            # We can use a set of invalid AP ids for faster updates
            invalid_aps = set()
            for ap_id in range(num_aps):
                if not is_ap_valid(colors, ap_id):
                    invalid_aps.add(ap_id)
            
            if len(invalid_aps) == 0:
                best_colors = colors[:]
                best_k = k
                found = True
                break
                
            # Local Search
            # We will try to reduce the number of invalid APs
            # Move: pick a random invalid AP, pick a random position in it, change its color
            # To make it faster, we can pick a position that is in many invalid APs?
            # Or just random position in a random invalid AP.
            
            stuck_counter = 0
            max_stuck = 10000
            current_invalid_count = len(invalid_aps)
            
            while time.time() < deadline and current_invalid_count > 0:
                if len(invalid_aps) == 0:
                    break
                    
                # Pick a random invalid AP
                # Converting set to list every time is slow. We can maintain a list of invalid APs.
                # Let's switch to a list of invalid APs for O(1) random access.
                # But removing from list is O(N).
                # Given small N (2667), set is fine for membership, but random choice is O(N) if we convert.
                # Let's use a list `invalid_list` and swap-delete for O(1) removal.
                
                if restart == 0 and not hasattr(construct, '_invalid_list_cache'):
                     # We'll just implement a simple list-based approach inside the loop for clarity, 
                     # but optimizing for speed is key.
                     pass
                
                # For simplicity and speed in this context, let's just pick a random index from the set
                # If the set is large, this is slow.
                # Let's maintain a list of invalid AP ids.
                if not hasattr(construct, '_use_list'):
                    construct._use_list = True
                
                # Re-initialize invalid list for this restart if not already done
                # Actually, let's just use the set for now and hope it's fast enough for < 10000 steps.
                # 2667 items in a set, random.choice(list(set)) is O(2667).
                # 10000 steps * 2667 ops = 26M ops. Might be tight in Python.
                
                # Better: maintain a list of invalid APs.
                # Let's refactor the inner loop to use a list.
                
                # We'll break out of this restart if it takes too long
                if current_invalid_count > 0:
                    # Pick a random invalid AP
                    # To avoid O(N) conversion, we can keep a list `invalid_list` in sync with `invalid_aps`.
                    # But managing sync is error prone.
                    # Let's try a different strategy: Randomly pick a position, then check APs containing it.
                    
                    pos = random.randint(0, N - 1)
                    # Get APs containing pos
                    candidate_aps = ap_indices_for_pos[pos]
                    
                    # Filter to only invalid ones
                    invalid_candidates = [ap_id for ap_id in candidate_aps if ap_id in invalid_aps]
                    
                    if not invalid_candidates:
                        stuck_counter += 1
                        if stuck_counter > max_stuck:
                            break
                        continue
                    else:
                        stuck_counter = 0
                    
                    # Pick one
                    target_ap = random.choice(invalid_candidates)
                    
                    # Try changing color of 'pos'
                    old_color = colors[pos]
                    # Try all other k-1 colors
                    best_new_color = None
                    min_violations_after = current_invalid_count + 100 # Large number
                    
                    # We need to evaluate the change efficiently.
                    # Changing colors[pos] affects all APs in candidate_aps.
                    # We can simulate the change for each new color.
                    
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        
                        # Simulate
                        # Count how many APs in candidate_aps become invalid/valid
                        # We need to know the state of neighbors.
                        
                        # This is getting complex. Let's just try the move and count the difference.
                        # Since k is small (3-10), we can just apply the change, count violations in affected APs, 
                        # and revert if worse.
                        
                        # Optimization: Only count violations for APs in candidate_aps.
                        # Total violations = (Total violations - violations in candidate_aps) + new violations in candidate_aps
                        # We don't have "violations in candidate_aps" readily available.
                        
                        # Alternative: Just re-count total violations? No, too slow.
                        
                        # Let's stick to the simple heuristic:
                        # 1. Change color
                        # 2. Update invalid_aps set for affected APs
                        # 3. If size decreased, keep. Else revert.
                        
                        colors[pos] = new_color
                        
                        new_invalid_count = current_invalid_count
                        # Update invalid_aps
                        # We need to know which APs were invalid before.
                        # Let's just remove all candidate_aps from invalid_aps, then re-add invalid ones.
                        # This is O(deg(pos)).
                        
                        # Save old state of candidate_aps
                        old_states = {}
                        for ap_id in candidate_aps:
                            if ap_id in invalid_aps:
                                old_states[ap_id] = True
                                invalid_aps.remove(ap_id)
                            else:
                                old_states[ap_id] = False
                                
                        # Check new states
                        for ap_id in candidate_aps:
                            if not is_ap_valid(colors, ap_id):
                                invalid_aps.add(ap_id)
                                
                        new_invalid_count = len(invalid_aps)
                        
                        if new_invalid_count < current_invalid_count:
                            # Accept move
                            best_new_color = new_color
                            current_invalid_count = new_invalid_count
                            break
                        else:
                            # Revert
                            colors[pos] = old_color
                            # Restore invalid_aps
                            for ap_id, was_invalid in old_states.items():
                                if was_invalid:
                                    invalid_aps.add(ap_id)
                                else:
                                    if ap_id in invalid_aps:
                                        invalid_aps.remove(ap_id)
                            # Revert current_invalid_count
                            current_invalid_count = len(invalid_aps)
                            
                    if best_new_color is None:
                        # No improving move found for this pos
                        # Try another pos? Or break?
                        # For simplicity, break to next restart or try random move.
                        # Let's try a random move to escape local minima
                        if random.random() < 0.1:
                            # Random move
                            new_color = random.randint(0, k - 1)
                            if new_color != colors[pos]:
                                colors[pos] = new_color
                                # Update invalid_aps
                                for ap_id in candidate_aps:
                                    if ap_id in invalid_aps:
                                        invalid_aps.remove(ap_id)
                                for ap_id in candidate_aps:
                                    if not is_ap_valid(colors, ap_id):
                                        invalid_aps.add(ap_id)
                                current_invalid_count = len(invalid_aps)
                        else:
                            break # Stuck, restart
                
            if current_invalid_count == 0:
                best_colors = colors[:]
                best_k = k
                found = True
                break
                
        if found:
            break
            
    if best_colors is None:
        # Fallback: return a greedy coloring if no solution found in time
        # This ensures we return something valid (or at least a coloring)
        # The greedy one from seed is better than random garbage
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                    and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                    colors.append(color)
                    break
        return colors
        
    return best_colors
