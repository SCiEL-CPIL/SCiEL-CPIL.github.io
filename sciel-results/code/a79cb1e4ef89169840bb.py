import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin to be safe
    
    n = 128
    
    # Precompute all 4-term APs
    # A 4-term AP is (a, a+d, a+2d, a+3d)
    # d >= 1
    # a+3d <= n => a <= n - 3d
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            idx_to_aps[idx].append(i)

    def is_valid_ap(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Check if at least 3 distinct colors
        # If len(set) >= 3, valid
        # If len(set) < 3, invalid
        return len({c0, c1, c2, c3}) >= 3

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors."""
        bad = 0
        for ap in aps:
            if not is_valid_ap(colors, ap):
                bad += 1
        return bad

    # First, run the seed's greedy algorithm to get a baseline valid solution
    # This ensures we always have a valid solution to return
    base_colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            # Check all APs ending at x-1
            valid = True
            for d in range(1, (x-1)//3+1):
                # AP: x-3d, x-2d, x-d, x
                # Indices: x-1-3d, x-1-2d, x-1-d, x-1
                c1 = base_colors[x-1-3*d]
                c2 = base_colors[x-1-2*d]
                c3 = base_colors[x-1-d]
                c4 = color
                if len({c1, c2, c3, c4}) < 3:
                    valid = False
                    break
            if valid:
                base_colors.append(color)
                found = True
                break
        if not found:
            # Should not happen if 128 colors are enough, but just in case
            base_colors.append(0)
    
    base_distinct = len(set(base_colors))
    best_solution = base_colors
    best_distinct = base_distinct
    
    # Try to find a solution with fewer colors
    # We try k from 3 up to best_distinct - 1
    # If we find a valid solution with k colors, we can stop and return it
    # But we might find an even better one with fewer k? 
    # The task says "Minimize the number of labels actually used".
    # So we want the minimum k.
    
    # We will iterate k from 3 to best_distinct - 1
    # If we find a valid solution for k, that is the minimum possible (since we go up from 3).
    # Wait, is it possible that 3 colors work? 
    # The problem asks to minimize colors. 
    # If a solution exists with 3 colors, that's the best.
    
    # However, local search for small k might be hard.
    # We will try k=3, 4, ... up to best_distinct-1.
    # If we find a valid solution for k, we return it immediately? 
    # Yes, because we are iterating k in increasing order.
    # The first valid k found is the minimum.
    
    # But what if k=3 is impossible? We need to check.
    # We will run local search for each k.
    
    for k in range(3, best_distinct):
        if time.time() > deadline - 5:
            break
        
        # Local search for this k
        found_valid = False
        best_k_cost = float('inf')
        best_k_solution = None
        
        # Multiple restarts
        max_restarts = 20
        restart_count = 0
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization with k colors
            colors = np.random.randint(0, k, n)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                found_valid = True
                best_k_solution = colors.copy()
                break
                
            # Simulated Annealing
            T = 1.0
            no_improve = 0
            max_iters = 5000 # Adjusted for speed
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    found_valid = True
                    best_k_solution = colors.copy()
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    ap = aps[ap_idx]
                    # Check old state
                    old_valid = is_valid_ap(colors, ap)
                    
                    # Temporarily change
                    pos_in_ap = ap.index(i)
                    old_c = colors[ap[pos_in_ap]]
                    colors[ap[pos_in_ap]] = new_color
                    
                    # Check new state
                    new_valid = is_valid_ap(colors, ap)
                    
                    # Revert
                    colors[ap[pos_in_ap]] = old_c
                    
                    # Update delta
                    if not old_valid and new_valid:
                        delta -= 1
                    elif old_valid and not new_valid:
                        delta += 1
                    # else delta unchanged
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > 200:
                            break # Stuck, break to restart or cool down
                            
                # Cool down
                T *= 0.999
                if T < 0.01:
                    T = 0.01

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                found_valid = True
                break

        if found_valid:
            # We found a valid solution with k colors
            # Since we iterate k from 3 upwards, this is the minimum
            # Convert to list of ints
            return list(best_k_solution)
            
    # If no better solution found, return the baseline
    return base_colors
