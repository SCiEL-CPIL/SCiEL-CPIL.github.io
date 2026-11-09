import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin (evaluator kills at 90s)
    
    n = 128
    
    # Precompute all 4-term APs (nonconstant)
    # Positions are 0-indexed: a, a+d, a+2d, a+3d where a+3d < n
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]
    
    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorization."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        # A 4-tuple has <= 2 distinct colors iff it has >= 2 matching pairs
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        
        bad = matches >= 2
        return int(np.sum(bad))
    
    def compute_delta(colors, i, new_color):
        """Compute delta in bad count when changing colors[i] to new_color."""
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
    
    def simulated_annealing(k, initial_colors, T_start, T_min, cooling, max_iters, max_no_improve):
        """Run SA on colors with k colors."""
        colors = initial_colors.copy()
        current_cost = count_bad_colors(colors)
        
        if current_cost == 0:
            return colors, 0
        
        T = T_start
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            if current_cost == 0:
                break
                
            i = random.randint(0, n - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue
                
            delta = compute_delta(colors, i, new_color)
            
            if delta < 0:
                colors[i] = new_color
                current_cost += delta
                no_improve = 0
            else:
                if delta == 0 or random.random() < np.exp(-delta / T):
                    colors[i] = new_color
                    current_cost += delta
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return colors, current_cost
    
    def local_search(k, initial_colors, max_iters, max_no_improve):
        """Pure local search (greedy hill climbing)."""
        colors = initial_colors.copy()
        current_cost = count_bad_colors(colors)
        
        if current_cost == 0:
            return colors, 0
        
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            if current_cost == 0:
                break
                
            i = random.randint(0, n - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue
                
            delta = compute_delta(colors, i, new_color)
            
            if delta < 0:
                colors[i] = new_color
                current_cost += delta
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return colors, current_cost
    
    best_solution = None
    best_k = float('inf')
    best_cost = float('inf')
    
    # Try k = 3 first, then 4, 5...
    for k in range(3, 8):
        if time.time() > deadline - 10:
            break
        
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts with different initializations
        max_restarts = 30 if k == 3 else 20
        restart_count = 0
        
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            
            # Run SA
            result, cost = simulated_annealing(k, colors, 2.0, 0.05, 0.9995, 15000, 400)
            
            if cost < best_k_cost:
                best_k_cost = cost
                best_k_solution = result.copy()
                
                if cost == 0:
                    break
            
            # Also try local search from this result
            if cost > 0 and time.time() < deadline - 5:
                result2, cost2 = local_search(k, result, 10000, 300)
                if cost2 < best_k_cost:
                    best_k_cost = cost2
                    best_k_solution = result2.copy()
                    
                    if cost2 == 0:
                        break
        
        if best_k_cost < best_cost:
            best_cost = best_k_cost
            best_solution = best_k_solution
            best_k = k
            
        if best_k_cost == 0:
            # Found a valid solution with k colors
            return [int(c) for c in best_solution]
    
    # If we found a solution with some k > 3, return it
    if best_solution is not None and best_cost == 0:
        return [int(c) for c in best_solution]
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check APs where x is the last term: a, a+d, a+2d, x where x = a+3d
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                if a >= 0:
                    c1 = colors[a]
                    c2 = colors[a + d]
                    c3 = colors[a + 2 * d]
                    if len({color, c1, c2, c3}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    
    return colors
