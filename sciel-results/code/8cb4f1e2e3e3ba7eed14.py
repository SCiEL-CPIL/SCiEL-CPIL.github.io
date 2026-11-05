import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    k4_idx = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_indices)
        for e in e_indices:
            edge_to_k4s[e].append(k4_idx)
        k4_idx += 1
        
    num_k4s = len(k4s)
    
    # State variables
    # colors: list of int, length num_edges
    # k4_freq: list of dicts, length num_k4s. key: color, value: count
    # k4_distinct: list of int, length num_k4s
    # total_violation: int
    
    def init_state(colors):
        k4_freq = [{} for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        for k4_i in range(num_k4s):
            for e in k4_edges[k4_i]:
                c = colors[e]
                if c in k4_freq[k4_i]:
                    k4_freq[k4_i][c] += 1
                else:
                    k4_freq[k4_i][c] = 1
            k4_distinct[k4_i] = len(k4_freq[k4_i])
        total_violation = sum(max(0, 5 - d) for d in k4_distinct)
        return k4_freq, k4_distinct, total_violation

    def try_move(colors, k4_freq, k4_distinct, total_violation, e_idx, new_color):
        """
        Attempts to change edge e_idx to new_color.
        Returns (success, new_total_violation)
        If success, modifies colors, k4_freq, k4_distinct in place.
        """
        old_color = colors[e_idx]
        if old_color == new_color:
            return True, total_violation
            
        # Calculate delta violation
        # We need to check all K4s containing e_idx
        # For each K4, we update the frequency and distinct count
        # We store the changes to revert if we decide to reject
        
        changes = [] # (k4_idx, old_distinct, new_distinct)
        delta_v = 0
        feasible = True
        
        for k4_i in edge_to_k4s[e_idx]:
            freq = k4_freq[k4_i]
            
            # Remove old color
            count_old = freq.get(old_color, 0)
            if count_old == 1:
                del freq[old_color]
                distinct_change = -1
            else:
                freq[old_color] = count_old - 1
                distinct_change = 0
            
            # Add new color
            count_new = freq.get(new_color, 0)
            if count_new == 0:
                distinct_change += 1
            freq[new_color] = count_new + 1
            
            old_distinct = k4_distinct[k4_i]
            new_distinct = old_distinct + distinct_change
            k4_distinct[k4_i] = new_distinct
            
            old_viol = max(0, 5 - old_distinct)
            new_viol = max(0, 5 - new_distinct)
            delta_v += (new_viol - old_viol)
            
            changes.append((k4_i, old_distinct))
            
            # Early exit if violation increases too much? 
            # No, we need exact delta for acceptance criteria.
            
        new_total_violation = total_violation + delta_v
        colors[e_idx] = new_color
        return True, new_total_violation

    def revert_move(colors, k4_freq, k4_distinct, e_idx, old_color, changes):
        """
        Reverts the move on edge e_idx back to old_color.
        """
        new_color = colors[e_idx]
        
        for k4_i, old_distinct in changes:
            freq = k4_freq[k4_i]
            
            # Remove new color
            count_new = freq.get(new_color, 0)
            if count_new == 1:
                del freq[new_color]
                distinct_change = -1
            else:
                freq[new_color] = count_new - 1
                distinct_change = 0
                
            # Add old color
            count_old = freq.get(old_color, 0)
            if count_old == 0:
                distinct_change += 1
            freq[old_color] = count_old + 1
            
            # k4_distinct[k4_i] is already correct from the forward pass logic? 
            # Wait, in try_move we updated k4_distinct. 
            # In revert, we should restore it.
            # The logic above calculates the new state. 
            # We need to ensure k4_distinct is set to old_distinct.
            k4_distinct[k4_i] = old_distinct
            
        colors[e_idx] = old_color

    best_result = None
    best_colors_used = 100
    
    # Baseline fallback
    baseline_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Verify baseline is valid
    k4_freq, k4_distinct, total_violation = init_state(baseline_colors)
    if total_violation == 0:
        best_result = baseline_colors[:]
        best_colors_used = 17

    # Search for better solutions
    # Try C = 13, 14, 15, 16
    for target_C in [13, 14, 15, 16]:
        if time.time() > deadline - 5:
            break
            
        # If we already have a result with <= target_C colors, we don't need to search for this C
        # unless we want to find a strictly better one (lower C). 
        # Since we iterate C from low to high, if we find a solution for C, we can return 
        # or continue to see if we can find one for C-1? No, we started at 13.
        # If we find a solution for 13, that's optimal. Return.
        # If we find a solution for 14, we still want to try 13? No, we already tried 13.
        # So if we find a valid solution for target_C, and target_C < best_colors_used, update.
        # If target_C == 13, return immediately.
        
        restarts = 0
        max_restarts = 50 if target_C <= 14 else 20
        
        while restarts < max_restarts:
            if time.time() > deadline - 5:
                break
            restarts += 1
            
            # Initialize random coloring with target_C colors
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            k4_freq, k4_distinct, total_violation = init_state(colors)
            
            if total_violation == 0:
                best_result = colors[:]
                best_colors_used = target_C
                if target_C == 13:
                    return best_result
                break # Try next lower C? No, loop structure handles C. Break restart loop.
            
            # Local Search
            steps = 0
            max_steps = 5000
            no_improve_count = 0
            threshold = 500
            
            while steps < max_steps and time.time() < deadline - 5:
                steps += 1
                
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                # Pick random new color
                new_color = random.randint(0, target_C - 1)
                
                old_color = colors[e_idx]
                if old_color == new_color:
                    continue
                    
                # Try the move
                # We need to calculate delta without modifying state if we might reject?
                # Or modify and revert. Modify and revert is easier to implement correctly with freq dicts.
                
                # Save state for revert
                changes = []
                delta_v = 0
                feasible = True # Always feasible in this simple model
                
                for k4_i in edge_to_k4s[e_idx]:
                    freq = k4_freq[k4_i]
                    
                    # Remove old
                    c_old_count = freq.get(old_color, 0)
                    if c_old_count == 1:
                        del freq[old_color]
                        d_change = -1
                    else:
                        freq[old_color] = c_old_count - 1
                        d_change = 0
                        
                    # Add new
                    c_new_count = freq.get(new_color, 0)
                    if c_new_count == 0:
                        d_change += 1
                    freq[new_color] = c_new_count + 1
                    
                    old_distinct = k4_distinct[k4_i]
                    new_distinct = old_distinct + d_change
                    k4_distinct[k4_i] = new_distinct
                    
                    old_viol = max(0, 5 - old_distinct)
                    new_viol = max(0, 5 - new_distinct)
                    delta_v += (new_viol - old_viol)
                    
                    changes.append((k4_i, old_distinct))
                    
                new_total_violation = total_violation + delta_v
                
                # Acceptance
                accepted = False
                if new_total_violation < total_violation:
                    accepted = True
                    no_improve_count = 0
                elif new_total_violation == total_violation:
                    # Accept sideways moves with some probability to escape plateaus
                    if random.random() < 0.5:
                        accepted = True
                    else:
                        # Revert
                        for k4_i, old_distinct in changes:
                            freq = k4_freq[k4_i]
                            # Revert freq changes
                            # This is tricky to do in place without re-calculating.
                            # Let's just revert by doing the reverse operation.
                            # Reverse of (old->new) is (new->old)
                            # But we already modified freq. 
                            # It's safer to just re-initialize freq for these K4s if revert is complex?
                            # No, let's just implement a clean revert.
                            
                            # Actually, the logic above modified freq. 
                            # To revert, we need to undo the specific changes.
                            # It's safer to just store the old freq state? Too expensive.
                            # Let's just re-calculate distinct and freq for the affected K4s from colors?
                            # That's 91 K4s * 6 edges = 546 ops. Very fast.
                            pass # Placeholder for revert logic
                else:
                    # Revert
                    pass # Placeholder for revert logic
                
                if accepted:
                    colors[e_idx] = new_color
                    total_violation = new_total_violation
                    if total_violation == 0:
                        break
                else:
                    # Revert changes
                    for k4_i, old_distinct in changes:
                        # Re-calculate freq and distinct for this K4 from current colors
                        # This is robust and fast enough for 91 K4s
                        freq = {}
                        for e in k4_edges[k4_i]:
                            c = colors[e]
                            freq[c] = freq.get(c, 0) + 1
                        k4_freq[k4_i] = freq
                        k4_distinct[k4_i] = len(freq)
                    # colors[e_idx] is already old_color, so no change needed there.
                    # But wait, did we change colors[e_idx] in the 'try' phase?
                    # No, we only updated k4_freq and k4_distinct. colors[e_idx] was NOT updated yet.
                    # So we just need to restore k4_freq and k4_distinct.
                    # The re-calculation above does that.
                    
                    no_improve_count += 1
                    if no_improve_count > threshold:
                        break
            
            if total_violation == 0:
                best_result = colors[:]
                best_colors_used = target_C
                if target_C == 13:
                    return best_result
                break # Found valid for this C, try lower C? No, we are iterating C upwards. 
                # If we found 14, we should have tried 13 already. 
                # So if we are at 14 and found a solution, we update best. 
                # We don't need to restart for 14, but we might want to try 13 again? 
                # The loop `for target_C` handles 13 first. If 13 failed, we go to 14.
                # If 14 succeeds, we update best. We can break the restart loop.
                break

    if best_result is None:
        # Fallback to baseline
        return baseline_colors
    else:
        return best_result

# Ensure the function is named construct and returns the list
# The code above defines construct().
