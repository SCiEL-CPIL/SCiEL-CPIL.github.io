import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Convert to numpy arrays for faster access
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

    def compute_delta(colors, i, old_color, new_color):
        """Compute delta in bad count when changing colors[i] from old_color to new_color."""
        ap_indices = idx_to_aps_arr[i]
        if len(ap_indices) == 0:
            return 0
        
        # Get the relevant APs
        sub_aps = aps[ap_indices]  # shape (m, 4)
        
        # Get current colors for these APs
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        # Determine which position in each AP is index i
        pos_mask_0 = (sub_aps[:, 0] == i)
        pos_mask_1 = (sub_aps[:, 1] == i)
        pos_mask_2 = (sub_aps[:, 2] == i)
        pos_mask_3 = (sub_aps[:, 3] == i)
        
        # Old bad status
        old_matches = ((c0 == c1).astype(np.int8) + 
                       (c0 == c2).astype(np.int8) + 
                       (c0 == c3).astype(np.int8) + 
                       (c1 == c2).astype(np.int8) + 
                       (c1 == c3).astype(np.int8) + 
                       (c2 == c3).astype(np.int8))
        old_bad = old_matches >= 2
        
        # New colors: replace the position corresponding to i with new_color
        nc0 = np.where(pos_mask_0, new_color, c0)
        nc1 = np.where(pos_mask_1, new_color, c1)
        nc2 = np.where(pos_mask_2, new_color, c2)
        nc3 = np.where(pos_mask_3, new_color, c3)
        
        new_matches = ((nc0 == nc1).astype(np.int8) + 
                       (nc0 == nc2).astype(np.int8) + 
                       (nc0 == nc3).astype(np.int8) + 
                       (nc1 == nc2).astype(np.int8) + 
                       (nc1 == nc3).astype(np.int8) + 
                       (nc2 == nc3).astype(np.int8))
        new_bad = new_matches >= 2
        
        return int(np.sum(new_bad) - np.sum(old_bad))

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Local search for this k
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts with increasing effort
        max_restarts = 80
        restart_count = 0
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing with adaptive parameters
            T = 3.0
            T_min = 0.01
            cooling = 0.9997
            no_improve = 0
            max_no_improve = 800
            max_iters = 30000
            iter_count = 0
            
            for iter_count in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = compute_delta(colors, i, old_color, new_color)
                
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
                        if no_improve > max_no_improve:
                            break
                            
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        # Final polish: exact local search (only accept improvements)
        if best_k_solution is not None and best_k_cost > 0 and time.time() < deadline - 2:
            colors = best_k_solution.copy()
            current_cost = best_k_cost
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(n):
                    old_color = colors[i]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        delta = compute_delta(colors, i, old_color, new_color)
                        if delta < 0:
                            colors[i] = new_color
                            current_cost += delta
                            improved = True
                            break
                    if time.time() > deadline - 2:
                        break
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()

        if best_k_cost == 0:
            # Found a valid solution with k colors
            # Convert to list of ints
            return [int(c) for c in best_k_solution]
            
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
