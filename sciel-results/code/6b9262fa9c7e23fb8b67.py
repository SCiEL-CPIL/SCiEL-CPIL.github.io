"""Improved program using multi-start simulated annealing with diverse initializations."""
import numpy as np
import random
import time

def count_mono_4ap(colors, n=256):
    """Count monochromatic 4-term arithmetic progressions in colors[0..n-1]."""
    count = 0
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            c0 = colors[a]
            if c0 == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    
    # Precompute APs and position mapping for fast delta calculation
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    pos_to_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d_pos) in enumerate(ap_list):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d_pos].append(idx)

    def make_thue_morse():
        """Thue-Morse sequence: t(n) = popcount(n) % 2."""
        colors = []
        for i in range(n):
            # i is 0-indexed here, corresponding to integer i+1 in 1..n
            # Thue-Morse is usually defined on non-negative integers.
            # Let's use the standard t(i) for i in 0..n-1
            colors.append(bin(i).count('1') % 2)
        return colors

    def make_block_coloring(block_size):
        colors = []
        for i in range(n):
            colors.append((i // block_size) % 2)
        return colors
    
    def make_mod_coloring(mod):
        colors = []
        for i in range(n):
            colors.append((i % mod) % 2)
        return colors

    def make_xor_pattern(x):
        """Color based on (i // x) % 2 XOR (i % x) // (x//2) ... just simple parity mix."""
        colors = []
        for i in range(n):
            colors.append((i // x) % 2)
        return colors

    # Generate diverse candidates
    candidates = []
    
    # Thue-Morse and variations
    candidates.append(make_thue_morse())
    # Inverted Thue-Morse
    tm = make_thue_morse()
    candidates.append([1 - c for c in tm])
    
    # Block sizes
    for bs in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]:
        candidates.append(make_block_coloring(bs))
        
    # Mod-based
    for mod in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]:
        candidates.append(make_mod_coloring(mod))
        
    # Random initializations
    for _ in range(30):
        candidates.append([random.randint(0, 1) for _ in range(n)])
        
    # Find best initial coloring
    best_colors_init = None
    best_count_init = float('inf')
    for cand in candidates:
        c = count_mono_4ap(cand, n)
        if c < best_count_init:
            best_count_init = c
            best_colors_init = cand[:]
            
    # Multi-Start Simulated Annealing
    start_time = time.time()
    time_limit = 115  # seconds
    
    global_best_colors = best_colors_init[:]
    global_best_count = best_count_init
    
    # We will run several SA instances.
    # Each instance starts from a good candidate (either global best or a random perturbation of it)
    # and runs for a fixed time slice.
    
    num_starts = 10
    time_per_start = (time_limit - 5) / num_starts  # Leave 5s for overhead/final check
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Initialize current state
        # Strategy: Start from global best, but perturb it slightly to escape local minima
        if start_idx == 0:
            colors = global_best_colors[:]
        else:
            # Perturb the global best by flipping a few random bits
            colors = global_best_colors[:]
            num_flips = random.randint(5, 20)
            for _ in range(num_flips):
                pos = random.randint(0, n - 1)
                colors[pos] = 1 - colors[pos]
        
        current_count = count_mono_4ap(colors, n)
        
        # If this start is already worse, we might still try to improve it, 
        # but usually starting from best is good.
        
        # SA Parameters
        T = 5.0
        T_min = 0.001
        # Cooling rate should be such that we reach T_min in the allotted time
        # Rough estimate of iterations per second: ~10k-50k depending on overhead
        # Let's fix iterations based on time slice
        # Simpler: use a fixed number of iterations per start
        # 115s / 10 starts = 11.5s per start.
        # Let's do 50,000 iterations per start? 
        # Delta calc is O(degree). Degree is ~sum(1/d) * 4 ~ 4 * ln(n) ~ 24.
        # 50k * 24 ops ~ 1.2M ops. Very fast in Python? Maybe 0.1s.
        # So we can do more iterations. Let's do 200,000 iterations per start.
        max_iters = 200000
        
        cooling = (T_min / T) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit - 1:
                break
                
            pos = random.randint(0, n - 1)
            old_val = colors[pos]
            new_val = 1 - old_val
            
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                # Check mono before
                # Optimization: only check if the other 3 are same as old_val?
                # If other 3 are not all same, it can't be mono before (regardless of pos)
                # If other 3 are same as old_val, it IS mono before.
                # After flip:
                # If other 3 are same as new_val, it IS mono after.
                
                # Get other values
                # This is a bit tricky to optimize without indexing overhead.
                # Let's stick to the previous logic but make it tight.
                
                v_a = colors[a]
                v_b = colors[b]
                v_c = colors[c]
                v_d = colors[d_pos]
                
                # Identify which one is 'pos'
                # We know pos is one of a,b,c,d
                
                # Before state
                if v_a == v_b == v_c == v_d:
                    mono_before = True
                else:
                    mono_before = False
                    
                # After state
                # Only pos changes
                if a == pos:
                    v_a = new_val
                elif b == pos:
                    v_b = new_val
                elif c == pos:
                    v_c = new_val
                else: # d_pos == pos
                    v_d = new_val
                    
                if v_a == v_b == v_c == v_d:
                    mono_after = True
                else:
                    mono_after = False
                    
                if mono_before and not mono_after:
                    delta -= 1
                elif not mono_before and mono_after:
                    delta += 1
            
            if delta <= 0:
                colors[pos] = new_val
                current_count += delta
                if current_count < global_best_count:
                    global_best_count = current_count
                    global_best_colors = colors[:]
            else:
                if random.random() < np.exp(-delta / T):
                    colors[pos] = new_val
                    current_count += delta
                    # Even if worse locally, it might lead to better global?
                    # We only update global best if it's actually better.
                    # So no update here for global_best.
            
            T *= cooling
            if T < T_min:
                T = T_min

    return global_best_colors
