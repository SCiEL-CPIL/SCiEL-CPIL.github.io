import numpy as np
import time
import random

def construct():
    n = 128
    deadline = time.time() + 55.0  # Safety margin
    
    # Precompute all non-constant 4-APs
    ap_indices = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            ap_indices.append([a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1])
            
    if not ap_indices:
        return [0] * n
        
    ap_matrix = np.array(ap_indices, dtype=np.int32)
    num_aps = ap_matrix.shape[0]
    
    # Precompute vertex to APs mapping
    vertex_to_aps = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        for v in ap_matrix[ap_idx]:
            vertex_to_aps[v].append(ap_idx)
            
    # Convert to numpy arrays for faster iteration if needed, 
    # but lists are fine for small counts
    vertex_to_aps_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in vertex_to_aps]

    def is_bad_tuple(c0, c1, c2, c3):
        # Returns True if the tuple has < 3 distinct colors
        # 1 distinct
        if c0 == c1 and c1 == c2 and c2 == c3:
            return True
        # 2 distinct
        # If c2 is c0 or c1, and c3 is c0 or c1, then all in {c0, c1}
        if (c2 == c0 or c2 == c1) and (c3 == c0 or c3 == c1):
            return True
        # If c1 is c0 or c2, and c3 is c0 or c2, then all in {c0, c2}
        if (c1 == c0 or c1 == c2) and (c3 == c0 or c3 == c2):
            return True
        # If c1 is c0 or c3, and c2 is c0 or c3, then all in {c0, c3}
        if (c1 == c0 or c1 == c3) and (c2 == c0 or c2 == c3):
            return True
        return False

    def compute_bad_mask(colors):
        bad_mask = np.zeros(num_aps, dtype=bool)
        for i in range(num_aps):
            ap = ap_matrix[i]
            if is_bad_tuple(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                bad_mask[i] = True
        return bad_mask

    def update_bad_mask_incremental(colors, bad_mask, v, new_color):
        # v is already set to new_color in colors
        affected_aps = vertex_to_aps_arrays[v]
        for ap_idx in affected_aps:
            ap = ap_matrix[ap_idx]
            if is_bad_tuple(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                bad_mask[ap_idx] = True
            else:
                bad_mask[ap_idx] = False

    # Try small number of colors
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Run multiple restarts
        start_k_time = time.time()
        max_time_per_k = 12.0  # 4 * 12 = 48s, leaving 7s margin
        
        while time.time() < start_k_time + max_time_per_k and time.time() < deadline:
            
            # Random initialization
            colors = np.random.randint(0, k, size=n, dtype=np.int32)
            bad_mask = compute_bad_mask(colors)
            bad_count = np.sum(bad_mask)
            
            if bad_count == 0:
                return colors.tolist()
            
            # Min-Conflicts Search
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
                
                # Pick a random vertex in this AP to change
                v = ap[random.randint(0, 3)]
                old_color = colors[v]
                
                # Try all colors for v
                best_delta = 0
                best_color = old_color
                
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    
                    # Apply
                    colors[v] = new_color
                    update_bad_mask_incremental(colors, bad_mask, v, new_color)
                    
                    # Count bads in affected APs
                    # We need to know how many bads changed.
                    # Simpler: just count total bads in affected APs
                    # But we don't have the old count easily.
                    # Let's just count the bads in the affected APs after the move.
                    # And compare to a baseline?
                    # Actually, Min-Conflicts usually counts the number of conflicts for the variable.
                    # Let's count conflicts for v.
                    
                    conflicts = 0
                    for ap_i in vertex_to_aps_arrays[v]:
                        if bad_mask[ap_i]:
                            conflicts += 1
                    
                    # Revert
                    colors[v] = old_color
                    update_bad_mask_incremental(colors, bad_mask, v, old_color)
                    
                    # We want to minimize conflicts.
                    # But we need to compare to the current conflicts for v.
                    # Let's compute current conflicts for v before the loop.
                    # This is inefficient.
                    
                    # Alternative: Just track the change in global bad_count?
                    # No, local is better.
                    
                    # Let's restructure:
                    # 1. Compute current conflicts for v.
                    # 2. For each new color, compute new conflicts for v.
                    # 3. Pick min.
                    
                    pass # Placeholder, see below

                # Restructured:
                current_conflicts = 0
                for ap_i in vertex_to_aps_arrays[v]:
                    if bad_mask[ap_i]:
                        current_conflicts += 1

                best_conflicts = current_conflicts
                best_color = old_color

                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    
                    colors[v] = new_color
                    update_bad_mask_incremental(colors, bad_mask, v, new_color)
                    
                    new_conflicts = 0
                    for ap_i in vertex_to_aps_arrays[v]:
                        if bad_mask[ap_i]:
                            new_conflicts += 1
                    
                    colors[v] = old_color
                    update_bad_mask_incremental(colors, bad_mask, v, old_color)
                    
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_color = new_color

                if best_color != old_color:
                    colors[v] = best_color
                    update_bad_mask_incremental(colors, bad_mask, v, best_color)
                    bad_count = np.sum(bad_mask)
                    stuck = 0
                    
                    if bad_count == 0:
                        return colors.tolist()
                else:
                    # No improvement
                    stuck += 1
                    if stuck > max_stuck:
                        # Random kick
                        num_kicks = random.randint(3, 10)
                        for _ in range(num_kicks):
                            v_kick = random.randint(0, n - 1)
                            old_c = colors[v_kick]
                            new_c = random.randint(0, k - 1)
                            if new_c != old_c:
                                colors[v_kick] = new_c
                                update_bad_mask_incremental(colors, bad_mask, v_kick, new_c)
                        bad_count = np.sum(bad_mask)
                        stuck = 0

    # Fallback
    return list(range(n))
