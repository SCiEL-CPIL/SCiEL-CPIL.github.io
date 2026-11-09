import numpy as np
import time
import random

def construct():
    n = 128
    deadline = time.time() + 55.0  # 55s safety margin within 60s limit
    
    # Precompute all non-constant 4-APs as index arrays
    # AP: (a, a+d, a+2d, a+3d) for a in 1..n, d >= 1, a+3d <= n
    ap_indices = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            # Convert to 0-indexed
            ap_indices.append([a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1])
    
    if not ap_indices:
        return [0] * n
    
    ap_matrix = np.array(ap_indices, dtype=np.int32)
    num_aps = ap_matrix.shape[0]
    
    # Precompute for each vertex, which APs it participates in
    vertex_to_aps = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        for v in ap_matrix[ap_idx]:
            vertex_to_aps[v].append(ap_idx)
    
    # Convert to numpy arrays for faster access
    vertex_to_aps_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in vertex_to_aps]
    
    # Fast check: given colors array, compute bad_mask for all APs
    def compute_bad_mask(colors):
        ap_colors = colors[ap_matrix]
        c0 = ap_colors[:, 0]
        c1 = ap_colors[:, 1]
        c2 = ap_colors[:, 2]
        c3 = ap_colors[:, 3]
        
        d1 = (c0 != c1).astype(np.int8)
        d2 = ((c0 != c2) & (c1 != c2)).astype(np.int8)
        d3 = ((c0 != c3) & (c1 != c3) & (c2 != c3)).astype(np.int8)
        
        distinct_count = 1 + d1 + d2 + d3
        return distinct_count < 3
    
    # Incremental update: when changing vertex v from old_color to new_color,
    # only update APs that contain v
    def update_bad_mask_incremental(colors, bad_mask, v, old_color, new_color):
        if old_color == new_color:
            return bad_mask, 0
        
        affected_aps = vertex_to_aps_arrays[v]
        if len(affected_aps) == 0:
            return bad_mask, 0
        
        delta_bad = 0
        for ap_idx in affected_aps:
            ap = ap_matrix[ap_idx]
            
            # Find position of v in the AP
            pos = -1
            for i in range(4):
                if ap[i] == v:
                    pos = i
                    break
            
            if pos == -1:
                continue
            
            # Get current colors (v already has new_color in colors array)
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Compute new distinct count
            new_colors = [c0, c1, c2, c3]
            new_distinct = len(set(new_colors))
            new_bad = new_distinct < 3
            
            # Compute old distinct count (v had old_color)
            old_colors = [c0, c1, c2, c3]
            old_colors[pos] = old_color
            old_distinct = len(set(old_colors))
            old_bad = old_distinct < 3
            
            if old_bad and not new_bad:
                delta_bad -= 1
            elif not old_bad and new_bad:
                delta_bad += 1
            
            # Update bad_mask
            if new_bad:
                bad_mask[ap_idx] = True
            else:
                bad_mask[ap_idx] = False
        
        return bad_mask, delta_bad
    
    # Initialize best result
    best_colors = None
    best_bad_count = num_aps + 1
    
    # Try small number of colors first
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        # Run multiple restarts
        for restart in range(5000):
            if time.time() > deadline:
                break
                
            # Random initialization
            colors = np.random.randint(0, k, size=n, dtype=np.int32)
            bad_mask = compute_bad_mask(colors)
            bad_count = np.sum(bad_mask)
            
            if bad_count < best_bad_count:
                best_bad_count = bad_count
                best_colors = colors.copy()
                
            if bad_count == 0:
                return colors.tolist()
            
            # Local search: hill climbing with random kicks
            stuck = 0
            max_stuck = 50
            while bad_count > 0 and time.time() < deadline:
                # Find a bad AP
                bad_indices = np.where(bad_mask)[0]
                if len(bad_indices) == 0:
                    break
                    
                # Pick a random bad AP
                ap_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                ap = ap_matrix[ap_idx]
                
                # Try changing each vertex in this AP to each color
                best_delta = 0
                best_move = None
                
                for v in ap:
                    old_color = colors[v]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        
                        # Apply change
                        colors[v] = new_color
                        new_bad_mask, delta = update_bad_mask_incremental(colors, bad_mask, v, old_color, new_color)
                        
                        # Revert
                        colors[v] = old_color
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (v, new_color)
                
                if best_move is not None and best_delta < 0:
                    v, new_color = best_move
                    old_c = colors[v]
                    colors[v] = new_color
                    bad_mask, _ = update_bad_mask_incremental(colors, bad_mask, v, old_c, new_color)
                    bad_count = np.sum(bad_mask)
                    stuck = 0
                    
                    if bad_count < best_bad_count:
                        best_bad_count = bad_count
                        best_colors = colors.copy()
                        
                    if bad_count == 0:
                        return colors.tolist()
                else:
                    # No improving move found, random kick
                    stuck += 1
                    if stuck > max_stuck:
                        # Kick: change 2-3 random vertices
                        num_kicks = random.randint(1, 3)
                        for _ in range(num_kicks):
                            v = random.randint(0, n - 1)
                            old_c = colors[v]
                            new_c = random.randint(0, k - 1)
                            if new_c != old_c:
                                colors[v] = new_c
                                bad_mask, _ = update_bad_mask_incremental(colors, bad_mask, v, old_c, new_c)
                        bad_count = np.sum(bad_mask)
                        stuck = 0
                        
                        if bad_count < best_bad_count:
                            best_bad_count = bad_count
                            best_colors = colors.copy()

    # Return best found
    if best_colors is not None:
        return best_colors.tolist()
    else:
        return [0] * n
