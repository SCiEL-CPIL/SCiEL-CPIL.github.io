import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute for each index, which APs contain it and at what position
    index_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for j in range(4):
            idx = aps[i, j]
            index_aps[idx].append((i, j))
            
    index_aps_arr = []
    for i in range(N):
        if index_aps[i]:
            arr = np.array(index_aps[i], dtype=np.int32)
            index_aps_arr.append(arr)
        else:
            index_aps_arr.append(np.array([], dtype=np.int32).reshape(0, 2))

    # Precompute degree (number of APs each index is in)
    degrees = np.array([len(index_aps[i]) for i in range(N)], dtype=np.int32)
    
    def get_score(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(is_mono))

    def compute_delta_single(colors, idx):
        pairs = index_aps_arr[idx]
        if len(pairs) == 0:
            return 0
            
        ap_indices = pairs[:, 0]
        positions = pairs[:, 1]
        
        c0 = colors[aps[ap_indices, 0]]
        c1 = colors[aps[ap_indices, 1]]
        c2 = colors[aps[ap_indices, 2]]
        c3 = colors[aps[ap_indices, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))
        
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()
        
        mask0 = (positions == 0)
        mask1 = (positions == 1)
        mask2 = (positions == 2)
        mask3 = (positions == 3)
        
        n_c0[mask0] = 1 - n_c0[mask0]
        n_c1[mask1] = 1 - n_c1[mask1]
        n_c2[mask2] = 1 - n_c2[mask2]
        n_c3[mask3] = 1 - n_c3[mask3]
        
        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count

    def compute_delta_double(colors, i, j):
        pairs_i = index_aps_arr[i]
        pairs_j = index_aps_arr[j]
        
        if len(pairs_i) == 0 and len(pairs_j) == 0:
            return 0
            
        if len(pairs_i) == 0:
            return compute_delta_single(colors, j)
            
        if len(pairs_j) == 0:
            return compute_delta_single(colors, i)
            
        ap_indices = np.unique(np.concatenate([pairs_i[:, 0], pairs_j[:, 0]]))
        
        c0 = colors[aps[ap_indices, 0]]
        c1 = colors[aps[ap_indices, 1]]
        c2 = colors[aps[ap_indices, 2]]
        c3 = colors[aps[ap_indices, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))
        
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()
        
        pos_i_map = np.full(len(ap_indices), -1, dtype=np.int32)
        pos_j_map = np.full(len(ap_indices), -1, dtype=np.int32)
        
        ap_to_local = {ap: idx for idx, ap in enumerate(ap_indices)}
        
        for ap_idx, pos in pairs_i:
            if ap_idx in ap_to_local:
                pos_i_map[ap_to_local[ap_idx]] = pos
                
        for ap_idx, pos in pairs_j:
            if ap_idx in ap_to_local:
                pos_j_map[ap_to_local[ap_idx]] = pos
                
        mask_i0 = (pos_i_map == 0)
        mask_i1 = (pos_i_map == 1)
        mask_i2 = (pos_i_map == 2)
        mask_i3 = (pos_i_map == 3)
        
        n_c0[mask_i0] = 1 - n_c0[mask_i0]
        n_c1[mask_i1] = 1 - n_c1[mask_i1]
        n_c2[mask_i2] = 1 - n_c2[mask_i2]
        n_c3[mask_i3] = 1 - n_c3[mask_i3]
        
        mask_j0 = (pos_j_map == 0)
        mask_j1 = (pos_j_map == 1)
        mask_j2 = (pos_j_map == 2)
        mask_j3 = (pos_j_map == 3)
        
        n_c0[mask_j0] = 1 - n_c0[mask_j0]
        n_c1[mask_j1] = 1 - n_c1[mask_j1]
        n_c2[mask_j2] = 1 - n_c2[mask_j2]
        n_c3[mask_j3] = 1 - n_c3[mask_j3]
        
        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count

    def greedy_polish(colors, current_score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_idx = -1
            
            for idx in range(N):
                if time.time() >= time_limit:
                    break
                delta = compute_delta_single(colors, idx)
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
            
            if best_idx != -1:
                colors[best_idx] = 1 - colors[best_idx]
                current_score += best_delta
                improved = True
                
            if not improved and time.time() < time_limit - 1.0:
                for _ in range(10):
                    if time.time() >= time_limit - 1.0:
                        break
                    i = np.random.randint(0, N)
                    j = np.random.randint(0, N)
                    if i == j:
                        continue
                    delta = compute_delta_double(colors, i, j)
                    if delta < 0:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        current_score += delta
                        improved = True
                        break
                        
        return current_score

    def iterated_local_search(initial_colors, time_limit):
        colors = initial_colors.copy()
        current_score = get_score(colors)
        
        # First, do a greedy polish
        current_score = greedy_polish(colors, current_score, time_limit)
        
        # Now do iterated local search with perturbations
        while time.time() < time_limit - 2.0:
            # Perturb: flip 3-5 random bits
            num_flips = np.random.randint(3, 6)
            flip_indices = np.random.choice(N, num_flips, replace=False)
            for idx in flip_indices:
                colors[idx] = 1 - colors[idx]
            
            perturbed_score = get_score(colors)
            
            # Polish the perturbed solution
            perturbed_score = greedy_polish(colors, perturbed_score, time_limit)
            
            # If we found something better, keep it; otherwise, keep the best
            # (This is a simplified ILS; we don't track separate best here,
            #  the caller will compare)
            
        return colors, current_score

    best_score = float('inf')
    best_colors = None
    
    # Strategy: Use more restarts with shorter SA, then strong polishing
    # The key insight from the observation is that ILS with 1-bit flips
    # worked better than SA with 2-bit flips
    
    num_restarts = 30
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
            
        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 5.0)
        
        # Random initial coloring
        colors = np.random.randint(0, 2, N).astype(np.int32)
        current_score = get_score(colors)
        
        # Short SA phase to get into a good basin
        T = 10.0
        cooling_factor = 0.999
        min_T = 0.5
        
        sa_deadline = restart_deadline - 2.0  # Reserve time for polishing
        
        while time.time() < sa_deadline and T > min_T:
            idx = np.random.randint(0, N)
            delta = compute_delta_single(colors, idx)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                colors[idx] = 1 - colors[idx]
                current_score += delta
                
            T *= cooling_factor
            
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
        # Strong greedy polish
        current_score = greedy_polish(colors, current_score, restart_deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()

    # Final global polish on best solution
    if best_colors is not None and time.time() < deadline - 2.0:
        colors = best_colors.copy()
        current_score = get_score(colors)
        
        # Multiple rounds of greedy polish with perturbations
        for _ in range(5):
            if time.time() >= deadline - 2.0:
                break
            
            current_score = greedy_polish(colors, current_score, deadline - 2.0)
            
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
            
            # Try a small perturbation and polish again
            if time.time() < deadline - 3.0:
                num_flips = np.random.randint(2, 4)
                flip_indices = np.random.choice(N, num_flips, replace=False)
                for idx in flip_indices:
                    colors[idx] = 1 - colors[idx]
                
                current_score = get_score(colors)
                current_score = greedy_polish(colors, current_score, deadline - 2.0)
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()

    if best_colors is None:
        return [0] * N
        
    return [int(c) for c in best_colors]
