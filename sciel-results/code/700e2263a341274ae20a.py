import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin (evaluator kills at 90s)
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with d >= 1
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Map each index to the APs it belongs to
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]
    
    def count_bad(colors):
        """Count APs with <= 2 distinct colors."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        
        return int(np.sum(matches >= 2))
    
    def compute_delta(colors, i, new_color):
        """Delta in bad count when changing colors[i] to new_color."""
        ap_indices = idx_to_aps_arr[i]
        if len(ap_indices) == 0:
            return 0
        
        sub_aps = aps[ap_indices]
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        pos0 = (sub_aps[:, 0] == i)
        pos1 = (sub_aps[:, 1] == i)
        pos2 = (sub_aps[:, 2] == i)
        pos3 = (sub_aps[:, 3] == i)
        
        old_matches = ((c0 == c1).astype(np.int8) + 
                       (c0 == c2).astype(np.int8) + 
                       (c0 == c3).astype(np.int8) + 
                       (c1 == c2).astype(np.int8) + 
                       (c1 == c3).astype(np.int8) + 
                       (c2 == c3).astype(np.int8))
        old_bad = old_matches >= 2
        
        nc0 = np.where(pos0, new_color, c0)
        nc1 = np.where(pos1, new_color, c1)
        nc2 = np.where(pos2, new_color, c2)
        nc3 = np.where(pos3, new_color, c3)
        
        new_matches = ((nc0 == nc1).astype(np.int8) + 
                       (nc0 == nc2).astype(np.int8) + 
                       (nc0 == nc3).astype(np.int8) + 
                       (nc1 == nc2).astype(np.int8) + 
                       (nc1 == nc3).astype(np.int8) + 
                       (nc2 == nc3).astype(np.int8))
        new_bad = new_matches >= 2
        
        return int(np.sum(new_bad) - np.sum(old_bad))
    
    # Try k = 3 first (minimum possible)
    best_solution = None
    best_cost = float('inf')
    
    for k in range(3, 7):
        if time.time() > deadline - 15:
            break
        
        best_k_solution = None
        best_k_cost = float('inf')
        
        max_restarts = 80 if k == 3 else 40
        restart_count = 0
        
        while time.time() < deadline - 8 and restart_count < max_restarts:
            restart_count += 1
            
            if k == 3:
                # Try multiple initialization strategies for k=3
                if restart_count % 4 == 1:
                    # Random
                    colors = np.random.randint(0, k, n).astype(np.int32)
                elif restart_count % 4 == 2:
                    # Periodic pattern
                    colors = np.zeros(n, dtype=np.int32)
                    for i in range(n):
                        colors[i] = i % 3
                elif restart_count % 4 == 3:
                    # Block pattern
                    colors = np.zeros(n, dtype=np.int32)
                    block_size = max(1, n // 6)
                    for i in range(n):
                        colors[i] = (i // block_size) % 3
                else:
                    # Greedy-like with randomization
                    colors = np.zeros(n, dtype=np.int32)
                    for i in range(n):
                        if i == 0:
                            colors[i] = 0
                        else:
                            # Try to pick a color that doesn't create immediate bad APs
                            best_c = 0
                            best_bad = float('inf')
                            for c in range(k):
                                colors[i] = c
                                # Quick check of APs ending at i
                                bad = 0
                                for d in range(1, (i) // 3 + 1):
                                    if i - 3*d >= 0:
                                        a = i - 3*d
                                        ap_colors = [colors[a], colors[a+d], colors[a+2*d], colors[i]]
                                        if len(set(ap_colors)) <= 2:
                                            bad += 1
                                if bad < best_bad:
                                    best_bad = bad
                                    best_c = c
                            colors[i] = best_c
            else:
                colors = np.random.randint(0, k, n).astype(np.int32)
            
            current_cost = count_bad(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
            
            # Simulated Annealing
            T = 3.0
            T_min = 0.01
            cooling = 0.9998
            no_improve = 0
            max_no_improve = 800
            max_iters = 30000
            
            for _ in range(max_iters):
                if time.time() > deadline - 8:
                    break
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                
                delta = compute_delta(colors, i, new_color)
                
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > max_no_improve:
                            break
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
            
            if best_k_cost == 0:
                break
        
        if best_k_cost == 0:
            best_solution = best_k_solution
            best_cost = 0
            break
        
        if best_k_solution is not None and best_k_cost < best_cost:
            best_cost = best_k_cost
            best_solution = best_k_solution
    
    if best_solution is not None:
        return [int(c) for c in best_solution]
    
    # Fallback: return a valid (possibly non-optimal) coloring
    # Use greedy construction
    colors = [0]
    for x in range(2, 129):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                if a < 1:
                    break
                ap_colors = [colors[a - 1], colors[a - 1 + d], colors[a - 1 + 2 * d], color]
                if len(set(ap_colors)) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    
    return colors
