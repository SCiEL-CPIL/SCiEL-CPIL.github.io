"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # A 4-AP is (x, x+d, x+2d, x+3d) with 1 <= x and x+3d <= 256, d >= 1
    # 0-indexed: (x-1, x-1+d, x-1+2d, x-1+3d)
    aps = []
    for d in range(1, 86):  # d from 1 to 85 (since 3*d <= 255)
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    n_aps = len(aps)
    # Convert to numpy array for fast operations
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, precompute which APs it appears in and at which slot
    # pos_aps[pos] = list of (ap_index, slot) where slot is 0,1,2,3
    pos_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append((idx, 0))
        pos_aps[i1].append((idx, 1))
        pos_aps[i2].append((idx, 2))
        pos_aps[i3].append((idx, 3))
    
    def count_mono(colors):
        """Count monochromatic 4-APs given colors array of 0/1."""
        count = 0
        for i in range(n_aps):
            a, b, c, d = aps_arr[i]
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count
    
    # Precompute for each position: the list of AP indices it appears in
    pos_ap_indices = [[] for _ in range(256)]
    for idx in range(n_aps):
        i0, i1, i2, i3 = aps_arr[idx]
        pos_ap_indices[i0].append(idx)
        pos_ap_indices[i1].append(idx)
        pos_ap_indices[i2].append(idx)
        pos_ap_indices[i3].append(idx)
    
    # Convert to numpy arrays for vectorized delta computation
    pos_ap_indices_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_ap_indices]
    
    def calc_delta(pos, colors):
        """Calculate change in mono count if we flip position pos (vectorized)."""
        ap_idxs = pos_ap_indices_arr[pos]
        if len(ap_idxs) == 0:
            return 0
        # Get the APs involving this position
        sub_aps = aps_arr[ap_idxs]  # shape (m, 4)
        # For each AP, check if all 4 positions have the same color as pos
        # colors[sub_aps] gives shape (m, 4)
        sub_colors = colors[sub_aps]  # shape (m, 4)
        pos_color = colors[pos]
        # Count matches per AP
        matches = (sub_colors == pos_color).sum(axis=1)  # shape (m,)
        # matches == 4 means currently mono (all same as pos)
        # After flip: the other 3 positions need to match the new color
        # If matches == 4, currently mono, after flip not mono (delta -= 1)
        # If matches == 1, only pos matches, after flip all 3 others match new color (delta += 1)
        # Otherwise, no change
        delta = -np.sum(matches == 4) + np.sum(matches == 1)
        return int(delta)
    
    # Initialize with alternating blocks of 2 (same as seed)
    colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
    current_cost = count_mono(colors)
    best_colors = colors.copy()
    best_cost = current_cost
    
    # Simulated annealing with multiple restarts
    n_restarts = 3
    restart_time = 110.0 / n_restarts  # ~36s per restart
    
    for restart in range(n_restarts):
        if restart > 0:
            # Perturb best solution for new restart
            colors = best_colors.copy()
            # Randomly flip some positions
            n_flips = random.randint(10, 30)
            flip_positions = random.sample(range(256), n_flips)
            for p in flip_positions:
                colors[p] = 1 - colors[p]
            current_cost = count_mono(colors)
        
        t = 1.0
        cooling = 0.9995
        min_t = 0.005
        restart_deadline = time.time() + restart_time
        
        while time.time() < restart_deadline:
            # Check deadline frequently
            if time.time() > restart_deadline - 1:
                break
                
            # Pick a random position to flip
            pos = random.randint(0, 255)
            
            # Calculate delta
            delta = calc_delta(pos, colors)
            
            # Accept if improvement or with probability exp(-delta/t)
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / t)):
                colors[pos] = 1 - colors[pos]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
            
            # Cool down
            t *= cooling
            if t < min_t:
                t = min_t
    
    # Final polish: multiple passes of greedy local search
    for polish_pass in range(5):
        if time.time() > deadline - 2:
            break
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for pos in range(256):
                if time.time() > deadline - 2:
                    break
                delta = calc_delta(pos, best_colors)
                if delta < 0:
                    best_colors[pos] = 1 - best_colors[pos]
                    best_cost += delta
                    improved = True
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
