import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin to be safe
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)
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
        
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        
        bad = matches >= 2
        return int(np.sum(bad))

    def compute_delta_fast(colors, i, new_color):
        """Compute delta in bad count when changing colors[i] to new_color."""
        ap_idxs = idx_to_aps_arr[i]
        m = len(ap_idxs)
        if m == 0:
            return 0
        
        sub_aps = aps[ap_idxs]
        
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        # Determine position of i in each AP
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

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3 first, as it is the theoretical minimum
    for k in [3, 4, 5]:
        if time.time() > deadline - 5:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts of Simulated Annealing
        max_restarts = 200
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            T = 2.0
            T_min = 0.01
            cooling = 0.9999
            no_improve = 0
            max_no_improve = 500
            max_iters = 20000
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
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
                    
                delta = compute_delta_fast(colors, i, new_color)
                
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
            # Found a valid solution with k colors
            return [int(c) for c in best_k_solution]

    # Fallback: return the best solution found so far
    if best_k_solution is not None:
        return [int(c) for c in best_k_solution]
    
    # Absolute fallback: random
    return [int(c) for c in np.random.randint(0, 3, n)]

