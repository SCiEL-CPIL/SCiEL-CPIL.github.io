import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute 4-APs
    aps = []
    aps_containing = [[] for _ in range(128)]
    for d in range(1, 43):
        for a in range(1, 129 - 3*d):
            idx0 = a-1
            idx1 = a+d-1
            idx2 = a+2*d-1
            idx3 = a+3*d-1
            ap = (idx0, idx1, idx2, idx3)
            ap_idx = len(aps)
            aps.append(ap)
            for pos in ap:
                aps_containing[pos].append(ap_idx)
                
    n_aps = len(aps)
    # violations[i] = 1 if AP i is invalid (uses < 3 colors), 0 otherwise
    # We want to minimize sum(violations) to 0.
    
    best_colors = None
    best_violations = n_aps
    
    # Helper to check if AP is valid (>= 3 distinct colors)
    # Returns 0 if valid, 1 if invalid
    def ap_invalid(c0, c1, c2, c3):
        # Check if number of distinct colors < 3
        # Fast check:
        # If c0==c1==c2==c3: 1 color (invalid)
        # If c0==c1 and c2==c3 and c0!=c2: 2 colors (invalid)
        # If c0==c2 and c1==c3 and c0!=c1: 2 colors (invalid)
        # If c0==c3 and c1==c2 and c0!=c1: 2 colors (invalid)
        # Else: likely 3 or 4.
        # Let's just use set for clarity, it's O(1) for small sets.
        return 0 if len({c0, c1, c2, c3}) >= 3 else 1

    # Try to find a valid coloring
    # We will try k=3, 4, 5... but the problem says "Minimize the number of labels actually used".
    # The task asks to return the labels. The evaluator minimizes the number of distinct labels used.
    # So we should try to use as few colors as possible.
    # Let's start with k=3. If we can't find a valid one quickly, try k=4.
    
    for k in range(3, 8):
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize colors
            # Greedy init: try to avoid mono 3-APs
            colors = np.zeros(128, dtype=int)
            for i in range(128):
                # Try each color 0..k-1, pick one that doesn't create a mono 3-AP ending at i
                # 3-APs ending at i: (i-2d, i-d, i)
                # We just check recent history for simplicity in init
                valid_colors = []
                for c in range(k):
                    ok = True
                    # Check if assigning c to i creates a mono 3-AP
                    # Possible 3-APs ending at i:
                    # i-2, i-1, i (d=1)
                    if i >= 2 and colors[i-1] == c and colors[i-2] == c:
                        ok = False
                    # i-4, i-2, i (d=2)
                    if i >= 4 and colors[i-2] == c and colors[i-4] == c:
                        ok = False
                    # i-6, i-3, i (d=3)
                    if i >= 6 and colors[i-3] == c and colors[i-6] == c:
                        ok = False
                    # ... up to d such that i-2d >= 0
                    # For init, checking d=1,2,3 is often enough to get a "good" start.
                    # But let's be more thorough for init if k is small.
                    if ok:
                        valid_colors.append(c)
                
                if valid_colors:
                    colors[i] = random.choice(valid_colors)
                else:
                    # If all colors cause a mono 3-AP, pick random (we will fix in local search)
                    colors[i] = random.randint(0, k-1)
            
            # Calculate initial violations
            violations = np.zeros(n_aps, dtype=int)
            current_violations = 0
            for i, ap in enumerate(aps):
                inv = ap_invalid(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                violations[i] = inv
                current_violations += inv
            
            best_local_violations = current_violations
            best_local_colors = colors.copy()
            
            # Local Search
            # Simulated Annealing or Hill Climbing with restarts
            # We will use a simple hill climbing with random restarts of the "move" strategy
            # To avoid getting stuck, we allow some uphill moves (SA) or just random restarts.
            
            temp = 1.0
            # Steps per restart
            max_steps = 20000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                pos = random.randint(0, 127)
                old_color = colors[pos]
                
                # Choose a new color
                # Prefer colors that might fix violations
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                # Calculate delta in violations
                # Only APs containing 'pos' are affected
                delta = 0
                affected_aps = aps_containing[pos]
                
                # Store old violation states for revert if needed
                old_states = []
                
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    old_inv = violations[ap_idx]
                    old_states.append((ap_idx, old_inv))
                    
                    # Get new colors for this AP
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    # Replace color at pos
                    if ap[0] == pos: c0 = new_color
                    elif ap[1] == pos: c1 = new_color
                    elif ap[2] == pos: c2 = new_color
                    elif ap[3] == pos: c3 = new_color
                    
                    new_inv = ap_invalid(c0, c1, c2, c3)
                    
                    if old_inv == 0 and new_inv == 1:
                        delta += 1
                    elif old_inv == 1 and new_inv == 0:
                        delta -= 1
                        
                if delta < 0:
                    # Accept
                    colors[pos] = new_color
                    current_violations += delta
                    for ap_idx, old_inv in old_states:
                        # Recalculate new_inv to update violations array
                        ap = aps[ap_idx]
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        violations[ap_idx] = ap_invalid(c0, c1, c2, c3)
                    
                    if current_violations < best_local_violations:
                        best_local_violations = current_violations
                        best_local_colors = colors.copy()
                        
                        if best_local_violations == 0:
                            distinct = len(set(best_local_colors.tolist()))
                            if best_colors is None or distinct < len(set(best_colors.tolist())):
                                best_colors = best_local_colors.copy()
                                # If we found a valid one, we can try to reduce colors further?
                                # The loop over k handles that.
                                # Break out of step loop
                                break
                elif delta == 0:
                    # Accept (neutral)
                    colors[pos] = new_color
                    # Update violations array
                    for ap_idx, old_inv in old_states:
                        ap = aps[ap_idx]
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        violations[ap_idx] = ap_invalid(c0, c1, c2, c3)
                        
                else:
                    # Reject (uphill) - optionally accept with prob exp(-delta/temp)
                    # For simplicity and speed, let's stick to strict hill climbing 
                    # but rely on random restarts for escaping local minima.
                    # If we want SA:
                    # if random.random() < np.exp(-delta / temp):
                    #    ... accept ...
                    # else:
                    #    ... reject ...
                    pass
                
                temp *= 0.999
                
                if best_local_violations == 0:
                    break
                    
            if best_local_violations == 0:
                # Found a valid coloring with k colors
                # Update global best if it has fewer distinct colors (it should, as k increases)
                distinct = len(set(best_local_colors.tolist()))
                if best_colors is None or distinct < len(set(best_colors.tolist())):
                    best_colors = best_local_colors.copy()
                
                # If we found a valid one with k=3, that's optimal.
                # If k>3, we might still find a better one with fewer distinct colors?
                # No, we are iterating k from 3 upwards. The first valid one found with k=3 is best.
                # But we might not find one with k=3 quickly.
                # If we find one with k=4, it uses <= 4 colors.
                # We should continue to see if we can find one with fewer distinct colors?
                # The loop `for k in range(3, 8)` implies we try 3 first.
                # If we find a valid one with k=3, we can return immediately?
                # Yes, because we can't do better than 3 (min 3 colors required).
                if k == 3:
                    return best_colors.tolist()
                
        # If we didn't break early, continue to next k

    if best_colors is not None:
        return best_colors.tolist()
    else:
        # Fallback: Return a random coloring (will be invalid, but prevents crash)
        # Or a simple pattern
        return list(range(128))
