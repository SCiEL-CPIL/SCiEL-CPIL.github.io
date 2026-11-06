"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 128
    MAX_COLORS = 128 # Labels 0..127
    
    # Precompute 4-APs
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map position to taps
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    # Initial coloring: Hamming weight (popcount)
    # This is known to be valid for small N, let's check for N=128
    initial_colors = [bin(i).count('1') for i in range(N)]
    
    # Check validity of initial
    def is_valid(colors):
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True

    if not is_valid(initial_colors):
        # Fallback to greedy or random if popcount fails (unlikely for 128)
        # Let's use a random valid coloring if possible, or just random and let SA fix it
        initial_colors = [random.randint(0, 7) for _ in range(N)]

    # Simulated Annealing
    def sa_search(colors, time_limit):
        current_colors = colors[:]
        current_violations = 0
        current_num_colors = len(set(current_colors))
        
        # Precompute color counts
        color_counts = [0] * MAX_COLORS
        for c in current_colors:
            color_counts[c] += 1
            
        # Calculate initial violations
        for a, b, c, d in all_taps:
            if len({current_colors[a], current_colors[b], current_colors[c], current_colors[d]}) < 3:
                current_violations += 1
                
        best_colors = current_colors[:]
        best_cost = 10 * current_violations + current_num_colors
        
        T = 10.0
        cooling = 0.999
        
        start_t = time.time()
        
        while time.time() - start_t < time_limit:
            # Pick random position
            i = random.randint(0, N - 1)
            old_c = current_colors[i]
            
            # Pick random new color
            # Bias towards colors already in use to help minimize count?
            # Or just random 0..MAX_COLORS-1
            new_c = random.randint(0, MAX_COLORS - 1)
            
            if new_c == old_c:
                continue
                
            # Calculate delta violations
            delta_viol = 0
            for ti in pos_taps[i]:
                a, b, c, d = all_taps[ti]
                # Get colors of other 3 positions
                pts = [a, b, c, d]
                others = [p for p in pts if p != i]
                
                # Set of colors with old_c
                s_old = set()
                for p in others:
                    s_old.add(current_colors[p])
                s_old.add(old_c)
                v_old = 1 if len(s_old) < 3 else 0
                
                # Set of colors with new_c
                s_new = set()
                for p in others:
                    s_new.add(current_colors[p])
                s_new.add(new_c)
                v_new = 1 if len(s_new) < 3 else 0
                
                delta_viol += v_new - v_old
            
            # Calculate delta num colors
            old_count_old = color_counts[old_c]
            new_count_old = old_count_old - 1
            old_count_new = color_counts[new_c]
            new_count_new = old_count_new + 1
            
            # Number of colors used
            # We can track this incrementally or just compute if counts change
            # If old_count_old == 1, old_c disappears -> -1 color
            # If old_count_new == 0, new_c appears -> +1 color
            delta_nc = 0
            if old_count_old == 1:
                delta_nc -= 1
            if old_count_new == 0:
                delta_nc += 1
                
            delta_cost = 10 * delta_viol + delta_nc
            
            # Acceptance
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Apply move
                current_colors[i] = new_c
                color_counts[old_c] = new_count_old
                color_counts[new_c] = new_count_new
                current_violations += delta_viol
                current_num_colors += delta_nc
                
                current_cost = 10 * current_violations + current_num_colors
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = current_colors[:]
                    
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_colors

    # Run SA
    # Use multiple restarts if time permits
    best_overall = initial_colors
    best_overall_cost = 10 * (0 if is_valid(initial_colors) else sum(1 for a,b,c,d in all_taps if len({initial_colors[a], initial_colors[b], initial_colors[c], initial_colors[d]}) < 3)) + len(set(initial_colors))
    
    remaining_time = deadline - time.time()
    
    if remaining_time > 5:
        # Single long SA run
        result = sa_search(initial_colors, remaining_time - 2)
        
        # Check if better
        v = sum(1 for a,b,c,d in all_taps if len({result[a], result[b], result[c], result[d]}) < 3)
        nc = len(set(result))
        cost = 10 * v + nc
        
        if cost < best_overall_cost:
            best_overall = result
            best_overall_cost = cost
            
    return best_overall
# EVOLVE-BLOCK-END
