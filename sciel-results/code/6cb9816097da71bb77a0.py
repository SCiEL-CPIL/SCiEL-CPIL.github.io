import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = 128
    
    # Precompute all nonconstant 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, (i, j, k, l) in enumerate(aps):
        pos_aps[i].append(idx)
        pos_aps[j].append(idx)
        pos_aps[k].append(idx)
        pos_aps[l].append(idx)
    
    def compute_violations(colors):
        """Return list of AP indices that are violations"""
        viols = []
        for idx, (i, j, k, l) in enumerate(aps):
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                viols.append(idx)
        return viols
    
    def ap_is_violation(colors, idx):
        i, j, k, l = aps[idx]
        return len({colors[i], colors[j], colors[k], colors[l]}) < 3
    
    best_colors = None
    best_distinct = 128
    
    # Try different numbers of colors
    for max_k in range(4, 25):
        if time.time() > deadline - 15:
            break
        
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random initial coloring
            colors = [random.randint(0, max_k - 1) for _ in range(n)]
            viol_set = set(compute_violations(colors))
            
            if len(viol_set) == 0:
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors[:]
                    if best_distinct <= 4:
                        break
            else:
                # Local search with incremental updates
                cur_viol_count = len(viol_set)
                no_improve = 0
                max_steps = 3000
                
                for step in range(max_steps):
                    if time.time() > deadline - 5:
                        break
                    if cur_viol_count == 0:
                        distinct = len(set(colors))
                        if distinct < best_distinct:
                            best_distinct = distinct
                            best_colors = colors[:]
                        break
                    
                    # Pick a random position
                    pos = random.randint(0, n - 1)
                    old_color = colors[pos]
                    new_color = random.randint(0, max_k - 1)
                    if new_color == old_color:
                        continue
                    
                    # Compute change in violation count for APs involving pos
                    delta = 0
                    for ap_idx in pos_aps[pos]:
                        was_viol = ap_idx in viol_set
                        # Temporarily change
                        colors[pos] = new_color
                        is_viol = ap_is_violation(colors, ap_idx)
                        colors[pos] = old_color
                        
                        if was_viol and not is_viol:
                            delta -= 1
                        elif not was_viol and is_viol:
                            delta += 1
                    
                    if delta < 0 or (delta == 0 and random.random() < 0.3):
                        # Accept the change
                        colors[pos] = new_color
                        # Update viol_set
                        for ap_idx in pos_aps[pos]:
                            was_viol = ap_idx in viol_set
                            is_viol = ap_is_violation(colors, ap_idx)
                            if was_viol and not is_viol:
                                viol_set.discard(ap_idx)
                            elif not was_viol and is_viol:
                                viol_set.add(ap_idx)
                        cur_viol_count = len(viol_set)
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if no_improve > 500:
                        break
                
                if cur_viol_count == 0:
                    distinct = len(set(colors))
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = colors[:]
        
        if best_distinct <= max_k:
            # We found something with <= max_k colors, no need to try larger
            break
    
    if best_colors is None:
        # Fallback: greedy from seed
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3+1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                           for d in range(1, (x-1)//2+1))):
                    colors.append(color)
                    break
        return colors
    
    # Remap to 0..distinct-1
    color_map = {}
    result = []
    for c in best_colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        result.append(color_map[c])
    
    return result
