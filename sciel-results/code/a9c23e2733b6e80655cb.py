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
    # For fast delta calculation
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Convert to numpy arrays
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]
    
    # Precompute positions of each index in its APs
    # idx_to_aps_pos[i] is a list of (ap_idx, pos)
    idx_to_aps_pos = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for pos, idx in enumerate(ap):
            idx_to_aps_pos[idx].append((i, pos))
    
    idx_to_aps_pos_arr = []
    for i in range(n):
        if idx_to_aps_pos[i]:
            ap_idxs = np.array([x[0] for x in idx_to_aps_pos[i]], dtype=np.int32)
            positions = np.array([x[1] for x in idx_to_aps_pos[i]], dtype=np.int32)
            idx_to_aps_pos_arr.append((ap_idxs, positions))
        else:
            idx_to_aps_pos_arr.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        # Distinct count <= 2 iff:
        # (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        # OR all 4 same (covered by above)
        # OR 3 same 1 diff? e.g. A A A B. Distinct {A,B} = 2. Bad.
        # Let's just count distinct colors directly or use the pair matching logic.
        # Matches >= 2 pairs implies <= 2 distinct colors?
        # If 4 distinct: 0 matches.
        # If 3 distinct (A A B C): Matches: c0==c1 (if A,A). 1 match.
        # If 2 distinct (A A B B): Matches: c0==c1, c2==c3, c0!=c2... 
        #   Pairs: (0,1) match, (2,3) match. Total 2 matches.
        #   (A A A B): (0,1), (0,2), (1,2) match. Total 3 matches.
        #   (A B A B): (0,2), (1,3) match. Total 2 matches.
        #   (A B B A): (0,3), (1,2) match. Total 2 matches.
        #   (A A A A): 6 matches.
        # So, distinct <= 2 iff matches >= 2.
        
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
        ap_idxs, positions = idx_to_aps_pos_arr[i]
        m = len(ap_idxs)
        if m == 0:
            return 0
        
        sub_aps = aps[ap_idxs]  # shape (m, 4)
        
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        pos_mask_0 = (positions == 0)
        pos_mask_1 = (positions == 1)
        pos_mask_2 = (positions == 2)
        pos_mask_3 = (positions == 3)
        
        old_matches = ((c0 == c1).astype(np.int8) + 
                       (c0 == c2).astype(np.int8) + 
                       (c0 == c3).astype(np.int8) + 
                       (c1 == c2).astype(np.int8) + 
                       (c1 == c3).astype(np.int8) + 
                       (c2 == c3).astype(np.int8))
        old_bad = old_matches >= 2
        
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

    def local_search(k, max_iters=20000):
        """
        Run local search to minimize bad APs.
        Returns best coloring found.
        """
        colors = np.random.randint(0, k, n).astype(np.int32)
        current_cost = count_bad_colors(colors)
        
        if current_cost == 0:
            return colors, 0
            
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 5.0
        T_min = 0.05
        cooling = 0.9995
        
        no_improve = 0
        max_no_improve = 1000
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            if current_cost == 0:
                return colors, 0
                
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
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
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
                
        return best_colors, best_cost

    # Try to find a solution with k=3, then k=4, etc.
    # We want to minimize the number of colors used.
    # If we find a valid solution with k colors, the "value" is the number of distinct colors actually used in that solution.
    # Ideally, we use exactly k colors.
    
    best_global_solution = None
    best_global_value = 128 # Max possible
    
    # Try k=3
    if time.time() < deadline - 10:
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            sol, cost = local_search(3, max_iters=5000)
            if cost == 0:
                # Check distinct colors
                distinct = len(set(sol.tolist()))
                if distinct < best_global_value:
                    best_global_value = distinct
                    best_global_solution = sol
                # If we found a 3-coloring, that's likely optimal for this problem size?
                # W(3,4) is 76? No, W(3,4) is the van der Waerden number for 3 colors, length 4.
                # It is known that W(3,4) = 76? Or is it larger?
                # Actually, W(3,4) is the smallest N such that ANY 3-coloring has a mono 4-AP.
                # We are avoiding mono AND 2-colored APs. This is stricter.
                # So a 3-coloring might not exist for N=128.
                # If it does, value=3.
                break
    
    # Try k=4
    if best_global_value > 3 and time.time() < deadline - 5:
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            sol, cost = local_search(4, max_iters=5000)
            if cost == 0:
                distinct = len(set(sol.tolist()))
                if distinct < best_global_value:
                    best_global_value = distinct
                    best_global_solution = sol
                break

    # Try k=5
    if best_global_value > 4 and time.time() < deadline - 2:
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            sol, cost = local_search(5, max_iters=5000)
            if cost == 0:
                distinct = len(set(sol.tolist()))
                if distinct < best_global_value:
                    best_global_value = distinct
                    best_global_solution = sol
                break

    if best_global_solution is not None:
        return [int(c) for c in best_global_solution]
    
    # Fallback: Return a random valid-ish coloring or just random
    # If no valid coloring found, return a random one (will score 0 or bad)
    # But we must return a list of 128 ints.
    return [int(c) for c in np.random.randint(0, 5, 128)]
