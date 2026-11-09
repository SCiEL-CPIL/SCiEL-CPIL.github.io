import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs as numpy arrays for vectorization
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
    
    # Precompute position masks for each index in its APs
    # pos_masks[i] is a list of 4 boolean arrays (one per position) for APs containing i
    # Actually, we just need to know which column is i in each AP.
    # Let's precompute the position of i in each AP it belongs to.
    # pos_in_ap[i] is a list of integers (0, 1, 2, or 3) corresponding to idx_to_aps[i]
    pos_in_ap = [[] for _ in range(n)]
    for i in range(n):
        for ap_idx in idx_to_aps[i]:
            ap = aps_list[ap_idx]
            pos_in_ap[i].append(ap.index(i))
    pos_in_ap_arr = [np.array(lst, dtype=np.int8) for lst in pos_in_ap]

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorization."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        # An AP is bad if it uses <= 2 distinct colors
        # This happens when:
        # - all 4 same: c0==c1==c2==c3
        # - 2 colors with pattern aabb, abab, abba, baab, baba, baab
        # Simpler: count distinct values
        # A 4-tuple has <=2 distinct values iff:
        # (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        # covers all cases where at most 2 distinct values exist
        
        bad = ((c0 == c1) & (c2 == c3)) | \
              ((c0 == c2) & (c1 == c3)) | \
              ((c0 == c3) & (c1 == c2))
        return int(np.sum(bad))

    def compute_delta(colors, i, new_color):
        """Compute delta in bad count when changing colors[i] to new_color."""
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
        
        # Old bad status
        old_bad = ((c0 == c1) & (c2 == c3)) | \
                  ((c0 == c2) & (c1 == c3)) | \
                  ((c0 == c3) & (c1 == c2))
        
        # New colors: replace the position corresponding to i with new_color
        # We need to update the correct column based on pos_in_ap_arr[i]
        # Let's create copies of c0..c3 and modify them
        nc0 = c0.copy()
        nc1 = c1.copy()
        nc2 = c2.copy()
        nc3 = c3.copy()
        
        # This is inefficient if we do it per AP.
        # Instead, we can use np.where with masks.
        # But we need masks for each position.
        # Let's precompute masks for each index i? No, that's too much memory.
        # Let's just use the pos_in_ap_arr.
        
        # Actually, a simpler vectorized way:
        # For each AP in sub_aps, we know which position is i.
        # Let pos = pos_in_ap_arr[i] (shape m,)
        # We can create a 4D array? No.
        
        # Let's stick to the mask approach but optimize it.
        # pos_mask_0 = (pos == 0), etc.
        pos = pos_in_ap_arr[i]
        
        nc0 = np.where(pos == 0, new_color, c0)
        nc1 = np.where(pos == 1, new_color, c1)
        nc2 = np.where(pos == 2, new_color, c2)
        nc3 = np.where(pos == 3, new_color, c3)
        
        new_bad = ((nc0 == nc1) & (nc2 == nc3)) | \
                  ((nc0 == nc2) & (nc1 == nc3)) | \
                  ((nc0 == nc3) & (nc1 == nc2))
        
        return int(np.sum(new_bad) - np.sum(old_bad))

    def local_search(k, initial_colors=None, time_limit=5.0):
        """Run local search for k colors. Returns best solution found."""
        if initial_colors is None:
            colors = np.random.randint(0, k, n).astype(np.int32)
        else:
            colors = initial_colors.copy()
            
        current_cost = count_bad_colors(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        if current_cost == 0:
            return best_colors, 0

        T = 1.0
        T_min = 0.01
        cooling = 0.999
        no_improve = 0
        max_no_improve = 100
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit:
            if current_cost == 0:
                return colors, 0
            
            # Pick random index
            i = np.random.randint(n)
            old_color = colors[i]
            new_color = np.random.randint(0, k)
            if new_color == old_color:
                continue
                
            # Calculate delta cost
            delta = compute_delta(colors, i, new_color)
            
            if delta < 0:
                colors[i] = new_color
                current_cost += delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    colors[i] = new_color
                    current_cost += delta
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # Restart local search with new random init or break
                        break
                        
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min

        return best_colors, best_cost

    # Try k = 3, 4, 5...
    for k in range(3, 8):
        if time.time() > deadline - 15:
            break
            
        # Local search for this k
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        max_restarts = 20
        restart_count = 0
        while time.time() < deadline - 10 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            init_colors = np.random.randint(0, k, n).astype(np.int32)
            
            # Run local search with a fixed time limit per restart
            sol, cost = local_search(k, init_colors, time_limit=2.0)
            
            if cost < best_k_cost:
                best_k_cost = cost
                best_k_solution = sol
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            # Found a valid solution with k colors
            return [int(c) for c in best_k_solution]
            
    # Fallback: Greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
