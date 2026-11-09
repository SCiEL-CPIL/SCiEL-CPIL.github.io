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
    
    # Precompute for each AP which position each index occupies
    # ap_pos[i] = list of (ap_idx, pos_in_ap) for all APs containing i
    ap_pos = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps_list):
        for pos in range(4):
            ap_pos[ap[pos]].append((ap_idx, pos))
    
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
    
    def compute_best_move_fast(colors, i, k):
        """
        For position i, find the color c in [0, k) that minimizes the number of bad APs 
        among APs containing i. Uses precomputed AP positions for speed.
        Returns (best_color, delta_cost)
        """
        if not ap_pos[i]:
            return colors[i], 0
            
        ap_indices, positions = zip(*ap_pos[i]) if len(ap_pos[i]) > 1 else (ap_pos[i][0][0], ap_pos[i][0][1])
        ap_indices = np.array(ap_indices, dtype=np.int32)
        positions = np.array(positions, dtype=np.int32)
        
        sub_aps = aps[ap_indices]  # shape (m, 4)
        
        # Get current colors for these APs
        c0 = colors[sub_aps[:, 0]].copy()
        c1 = colors[sub_aps[:, 1]].copy()
        c2 = colors[sub_aps[:, 2]].copy()
        c3 = colors[sub_aps[:, 3]].copy()
        
        # Determine which column to modify based on position
        # positions tells us which column (0,1,2,3) index i occupies in each AP
        old_matches = ((c0 == c1).astype(np.int8) + 
                       (c0 == c2).astype(np.int8) + 
                       (c0 == c3).astype(np.int8) + 
                       (c1 == c2).astype(np.int8) + 
                       (c1 == c3).astype(np.int8) + 
                       (c2 == c3).astype(np.int8))
        old_bad = (old_matches >= 2).astype(np.int8)
        old_cost = int(np.sum(old_bad))
        
        best_delta = float('inf')
        best_color = colors[i]
        
        # Try each candidate color
        for c in range(k):
            if c == colors[i]:
                continue
                
            # Create modified copies
            nc0 = c0.copy()
            nc1 = c1.copy()
            nc2 = c2.copy()
            nc3 = c3.copy()
            
            # Modify the appropriate column
            mask0 = (positions == 0)
            mask1 = (positions == 1)
            mask2 = (positions == 2)
            mask3 = (positions == 3)
            
            if np.any(mask0):
                nc0[mask0] = c
            if np.any(mask1):
                nc1[mask1] = c
            if np.any(mask2):
                nc2[mask2] = c
            if np.any(mask3):
                nc3[mask3] = c
            
            new_matches = ((nc0 == nc1).astype(np.int8) + 
                           (nc0 == nc2).astype(np.int8) + 
                           (nc0 == nc3).astype(np.int8) + 
                           (nc1 == nc2).astype(np.int8) + 
                           (nc1 == nc3).astype(np.int8) + 
                           (nc2 == nc3).astype(np.int8))
            new_bad = (new_matches >= 2).astype(np.int8)
            new_cost = int(np.sum(new_bad))
            
            delta = new_cost - old_cost
            if delta < best_delta:
                best_delta = delta
                best_color = c
                
        return best_color, best_delta

    def simulated_annealing(colors, k, max_iters, time_limit):
        """Simulated annealing to find valid coloring."""
        colors = colors.copy()
        bad_indices = get_bad_aps(colors)
        current_cost = len(bad_indices)
        
        if current_cost == 0:
            return colors, 0
            
        T = 10.0
        cooling = 0.995
        best_colors = colors.copy()
        best_cost = current_cost
        no_improve = 0
        max_no_improve = 300
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
                
            if current_cost == 0:
                return colors, 0
                
            # Pick a random bad AP or random position
            if current_cost > 0 and random.random() < 0.8:
                # Pick from bad APs
                ap_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                ap = aps[ap_idx]
                pos_idx = random.randint(0, 3)
                i = ap[pos_idx]
            else:
                # Pick random position
                i = random.randint(0, n - 1)
                
            # Find best color for i
            best_color, delta = compute_best_move_fast(colors, i, k)
            
            # Simulated annealing: accept worse moves with probability exp(-delta/T)
            if delta < 0:
                colors[i] = best_color
                current_cost += delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors, 0
                bad_indices = get_bad_aps(colors)
                current_cost = len(bad_indices)
            else:
                # Accept with probability exp(-delta/T)
                if delta == 0 or random.random() < np.exp(-delta / T):
                    colors[i] = best_color
                    current_cost += delta
                    bad_indices = get_bad_aps(colors)
                    current_cost = len(bad_indices)
                    no_improve += 1
                else:
                    no_improve += 1
                    
            T *= cooling
            if T < 0.1:
                T = 0.1
                
            if no_improve > max_no_improve:
                # Restart with higher temperature
                T = 5.0
                no_improve = 0
                
        return best_colors, best_cost

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts with simulated annealing
        best_k_solution = None
        best_k_cost = float('inf')
        
        max_restarts = 30
        restart_count = 0
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            
            # Run simulated annealing
            sa_solution, sa_cost = simulated_annealing(
                colors, k, 
                max_iters=3000,
                time_limit=deadline - 5
            )
            
            if sa_cost < best_k_cost:
                best_k_cost = sa_cost
                best_k_solution = sa_solution.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            return [int(c) for c in best_k_solution]
            
        # Update global best
        if best_k_cost < best_cost:
            best_cost = best_k_cost
            best_solution = best_k_solution.copy()
            
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
