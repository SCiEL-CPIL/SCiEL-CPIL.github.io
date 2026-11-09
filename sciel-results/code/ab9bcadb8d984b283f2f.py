import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    # Precompute "others" for each position to speed up flip_delta
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p] # shape (K, 4)
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)

    # Count monochromatic APs for a coloring (0/1 array)
    def count_mono(color):
        cols = color[aps_arr]  # shape (M, 4)
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    # Compute delta in mono count if we flip position p
    def flip_delta(color, p):
        """Return change in mono count if we flip color[p]."""
        others = pos_others[p]
        if others.shape[0] == 0:
            return 0
        c_p = color[p]
        c_new = 1 - c_p
        # Get colors of the other 3 positions for all APs containing p
        others_colors = color[others] # shape (K, 3)
        is_mono_now = np.all(others_colors == c_p, axis=1)
        is_mono_new = np.all(others_colors == c_new, axis=1)
        return int(np.sum(is_mono_new) - np.sum(is_mono_now))
    
    # Thue-Morse initial coloring
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = bin(i).count('1') % 2
        return color
    
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color
    
    # Local search: hill climbing with first-improvement (randomized order)
    def local_search(color, max_iter=5000):
        color = color.copy()
        current_score = count_mono(color)
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            perm = np.random.permutation(n)
            for p in perm:
                d = flip_delta(color, p)
                if d < 0:
                    color[p] = 1 - color[p]
                    current_score += d
                    improved = True
                    break
            if not improved:
                break
        return color, current_score
    
    # Simulated annealing with restarts
    def simulated_annealing(color, max_time=30.0):
        color = color.copy()
        best_color = color.copy()
        best_score = count_mono(color)
        current_score = best_score
        
        T = 1.0
        cooling = 0.9998
        min_T = 0.01
        stuck_counter = 0
        no_improve_steps = 0
        max_no_improve = 5000
        
        end_time = time.time() + max_time
        
        while time.time() < end_time:
            pos = np.random.randint(0, n)
            d = flip_delta(color, pos)
            
            # Acceptance criterion
            accept = False
            if d < 0:
                accept = True
                stuck_counter = 0
                no_improve_steps = 0
            elif d == 0:
                accept = True
                stuck_counter = 0
            else:
                if T > 0.05:
                    prob = math.exp(-d / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    accept = True
                    stuck_counter = 0
            
            if accept:
                color[pos] = 1 - color[pos]
                current_score += d
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                stuck_counter += 1
                no_improve_steps += 1
            
            # Cooling
            T *= cooling
            if T < min_T:
                # Restart with perturbation
                T = 1.0
                stuck_counter = 0
                # Perturb: flip several random positions
                for _ in range(8):
                    p2 = np.random.randint(0, n)
                    d2 = flip_delta(color, p2)
                    if d2 <= 0 or (T > 0.1 and random.random() < 0.1):
                        color[p2] = 1 - color[p2]
                        current_score += d2
                        if current_score < best_score:
                            best_score = current_score
                            best_color = color.copy()
                            no_improve_steps = 0
            
            # If stuck too long, do a larger perturbation
            if stuck_counter > 2000:
                T = 1.5
                stuck_counter = 0
                for _ in range(12):
                    p2 = np.random.randint(0, n)
                    d2 = flip_delta(color, p2)
                    if d2 <= 0 or random.random() < 0.15:
                        color[p2] = 1 - color[p2]
                        current_score += d2
                        if current_score < best_score:
                            best_score = current_score
                            best_color = color.copy()
                            no_improve_steps = 0
            
            # If no improvement for a long time, restart from best_color with perturbation
            if no_improve_steps > max_no_improve:
                color = best_color.copy()
                current_score = best_score
                T = 1.0
                no_improve_steps = 0
                # Perturb
                for _ in range(6):
                    p2 = np.random.randint(0, n)
                    d2 = flip_delta(color, p2)
                    if d2 <= 0 or random.random() < 0.1:
                        color[p2] = 1 - color[p2]
                        current_score += d2
        
        return best_color, best_score
    
    # Generate diverse starting points
    startings = [thue_morse(n)]
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(8):
        startings.append(random_coloring())
    
    # Also try Thue-Morse with random perturbations
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=10, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # Phase 1: Run simulated annealing from diverse starting points
    for start_color in startings:
        if time.time() - start_time > deadline - 15:
            break
        
        # Run simulated annealing with time limit
        remaining = deadline - 15 - (time.time() - start_time)
        if remaining < 10:
            break
            
        color, score = simulated_annealing(start_color, max_time=min(remaining, 25.0))
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # Phase 2: Local search polish on the best result
    if best_overall_color is not None and time.time() - start_time < deadline - 5:
        color, score = local_search(best_overall_color, max_iter=10000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # Phase 3: Final simulated annealing on the best result if time permits
    if best_overall_color is not None and time.time() - start_time < deadline - 10:
        remaining = deadline - 10 - (time.time() - start_time)
        if remaining > 5:
            color, score = simulated_annealing(best_overall_color, max_time=remaining)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()
    
    # Phase 4: Final local search polish
    if best_overall_color is not None and time.time() - start_time < deadline - 2:
        color, score = local_search(best_overall_color, max_iter=5000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
