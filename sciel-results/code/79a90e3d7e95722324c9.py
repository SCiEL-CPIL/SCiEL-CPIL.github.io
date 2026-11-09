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
    
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]

    def get_bad_aps(colors):
        """Return indices of APs with <= 2 distinct colors."""
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
        return np.where(bad)[0]

    def compute_best_move(colors, i, k):
        """
        For position i, find the color c in [0, k) that minimizes the number of bad APs 
        among APs containing i.
        Returns (best_color, delta_cost)
        """
        ap_indices = idx_to_aps_arr[i]
        if len(ap_indices) == 0:
            return colors[i], 0
            
        sub_aps = aps[ap_indices]  # shape (m, 4)
        
        # Get current colors for these APs
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        # Determine which position in each AP is index i
        # We need to replace the value at position i with candidate colors and count badness.
        
        # Base badness count for the current state
        old_matches = ((c0 == c1).astype(np.int8) + 
                       (c0 == c2).astype(np.int8) + 
                       (c0 == c3).astype(np.int8) + 
                       (c1 == c2).astype(np.int8) + 
                       (c1 == c3).astype(np.int8) + 
                       (c2 == c3).astype(np.int8))
        old_bad = (old_matches >= 2).astype(np.int8)
        old_cost = np.sum(old_bad)
        
        best_delta = float('inf')
        best_color = colors[i]
        
        # Try each candidate color
        for c in range(k):
            if c == colors[i]:
                continue
                
            nc0 = np.where(sub_aps[:, 0] == i, c, c0)
            nc1 = np.where(sub_aps[:, 1] == i, c, c1)
            nc2 = np.where(sub_aps[:, 2] == i, c, c2)
            nc3 = np.where(sub_aps[:, 3] == i, c, c3)
            
            new_matches = ((nc0 == nc1).astype(np.int8) + 
                           (nc0 == nc2).astype(np.int8) + 
                           (nc0 == nc3).astype(np.int8) + 
                           (nc1 == nc2).astype(np.int8) + 
                           (nc1 == nc3).astype(np.int8) + 
                           (nc2 == nc3).astype(np.int8))
            new_bad = (new_matches >= 2).astype(np.int8)
            new_cost = np.sum(new_bad)
            
            delta = new_cost - old_cost
            if delta < best_delta:
                best_delta = delta
                best_color = c
                
        return best_color, best_delta

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Min-Conflicts / Greedy Repair Search
        best_k_solution = None
        best_k_cost = float('inf')
        
        max_restarts = 20
        restart_count = 0
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            bad_indices = get_bad_aps(colors)
            current_cost = len(bad_indices)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Iterative improvement
            max_iters = 5000
            no_improve = 0
            max_no_improve = 200
            
            for iter_count in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick a random bad AP
                ap_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                ap = aps[ap_idx]
                
                # Pick a random position in this AP
                pos_idx = random.randint(0, 3)
                i = ap[pos_idx]
                
                # Find best color for i
                best_color, delta = compute_best_move(colors, i, k)
                
                if delta < 0:
                    colors[i] = best_color
                    current_cost += delta
                    no_improve = 0
                    # Update bad_indices
                    # Recomputing is fast enough (~2700 ops)
                    bad_indices = get_bad_aps(colors)
                    current_cost = len(bad_indices)
                else:
                    # If no improvement, try a random move to escape local minima?
                    # Or just break if stuck.
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
                        
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                if best_k_cost == 0:
                    best_k_solution = colors.copy()
                else:
                    best_k_solution = colors.copy() # Keep best found so far for this k
            
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            return [int(c) for c in best_k_solution]
            
    # Fallback: return the best solution found for any k
    if best_solution is not None:
        return [int(c) for c in best_solution]
        
    # Absolute fallback: Greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
