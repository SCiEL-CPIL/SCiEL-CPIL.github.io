import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Precompute all 4-term arithmetic progressions as numpy arrays
    # A 4-AP: a, a+d, a+2d, a+3d with a >= 1, a+3d <= N, d >= 1
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    # Shape: (total_aps, 4)
    aps = np.array(ap_list, dtype=np.int32)
    
    # For each position, which APs contain it (for delta computation)
    # We'll use a different approach: for each position, store indices into aps
    pos_to_aps = [[] for _ in range(N)]
    for i in range(total_aps):
        for j in range(4):
            pos_to_aps[aps[i, j]].append(i)
    
    # Precompute for each position: the AP indices and the position within that AP
    # pos_ap_indices[p] = array of AP indices containing p
    # pos_ap_positions[p] = array of which position (0-3) p occupies in each AP
    pos_ap_indices = []
    pos_ap_positions = []
    for p in range(N):
        idxs = np.array(pos_to_aps[p], dtype=np.int32)
        positions = np.array([list(aps[i]).index(p) for i in pos_to_aps[p]], dtype=np.int8)
        pos_ap_indices.append(idxs)
        pos_ap_positions.append(positions)
    
    def count_mono(colors):
        """Count monochromatic 4-APs using numpy. colors is numpy array of 0/1."""
        # For each AP, check if all 4 positions have same color
        # colors[aps] has shape (total_aps, 4)
        c = colors[aps]  # shape (total_aps, 4)
        # All equal if max == min for each row
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())
    
    def compute_delta(colors, p):
        """Compute change in mono count if we flip position p."""
        old_color = colors[p]
        new_color = 1 - old_color
        ap_idxs = pos_ap_indices[p]
        if len(ap_idxs) == 0:
            return 0
        
        # Get all APs containing p
        sub_aps = aps[ap_idxs]  # shape (num_aps, 4)
        positions = pos_ap_positions[p]  # shape (num_aps,)
        
        # Current colors of all positions in these APs
        sub_colors = colors[sub_aps]  # shape (num_aps, 4)
        
        # Before flip: check if mono
        mono_before = (sub_colors.max(axis=1) == sub_colors.min(axis=1))
        
        # After flip: replace position p's color
        sub_colors_after = sub_colors.copy()
        for i in range(len(ap_idxs)):
            sub_colors_after[i, positions[i]] = new_color
        
        mono_after = (sub_colors_after.max(axis=1) == sub_colors_after.min(axis=1))
        
        delta = int(mono_after.sum()) - int(mono_before.sum())
        return delta
    
    def fast_compute_delta(colors, p):
        """Faster delta computation using precomputed structure."""
        old_color = colors[p]
        ap_idxs = pos_ap_indices[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0
        
        sub_aps = aps[ap_idxs]  # shape (n_aps, 4)
        positions = pos_ap_positions[p]  # shape (n_aps,)
        
        # Get colors of the other 3 positions in each AP
        # For each AP, we need to know the colors of the 3 positions that aren't p
        # sub_aps has shape (n_aps, 4), positions tells us which col is p
        sub_colors = colors[sub_aps]  # shape (n_aps, 4)
        
        # Before: check if all 4 are same
        mono_before = (sub_colors.max(axis=1) == sub_colors.min(axis=1))
        
        # After: flip position p in each AP
        # Create a copy and flip
        sub_colors_after = sub_colors.copy()
        rows = np.arange(n_aps)
        sub_colors_after[rows, positions] = 1 - old_color
        
        mono_after = (sub_colors_after.max(axis=1) == sub_colors_after.min(axis=1))
        
        return int(mono_after.sum()) - int(mono_before.sum())
    
    best_colors = None
    best_count = float('inf')
    
    # Helper to update best
    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()
    
    # Strategy: multiple restarts with simulated annealing + local search
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            # Alternating pairs (seed pattern)
            colors = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 1:
            # Random
            colors = np.random.randint(0, 2, size=N).astype(np.int8)
        elif restart == 2:
            # Thicker blocks: 000111000111...
            colors = np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 3:
            # Thicker blocks: 0000111100001111...
            colors = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8)
        else:
            if best_colors is not None:
                colors = best_colors.copy()
                # Perturb ~15% of bits
                n_flips = max(5, N // 7)
                flip_positions = np.random.choice(N, size=n_flips, replace=False)
                colors[flip_positions] ^= 1
            else:
                colors = np.random.randint(0, 2, size=N).astype(np.int8)
        
        current_count = count_mono(colors)
        update_best(colors, current_count)
        
        # Simulated annealing
        T = 3.0
        T_min = 0.05
        cooling = 0.9998
        max_sa_steps = 80000
        steps_done = 0
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick random position
            p = np.random.randint(0, N)
            
            # Compute delta
            delta = fast_compute_delta(colors, p)
            
            if delta <= 0:
                colors[p] ^= 1
                current_count += delta
                update_best(colors, current_count)
            else:
                # Accept with probability exp(-delta/T)
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        colors[p] ^= 1
                        current_count += delta
                # else reject (T too low)
            
            T = max(T_min, T * cooling)
            steps_done += 1
        
        # Local search: hill climbing
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for p in range(N):
                if time.time() > deadline - 3:
                    break
                delta = fast_compute_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)
        
        # Second pass of local search (might find more improvements)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for p in range(N):
                if time.time() > deadline - 2:
                    break
                delta = fast_compute_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
