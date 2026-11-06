import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    n = 128
    
    # Precompute all nonconstant 4-term APs
    aps = []
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 >= n:
                break
            aps.append((a, a + d, a + 2 * d, a3))
    
    # For each position, store indices of APs it participates in
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for p in ap:
            pos_to_aps[p].append(i)
    
    def greedy_assign(num_colors, seed=0):
        rng = random.Random(seed)
        colors = [-1] * n
        
        for pos in range(n):
            color_order = list(range(num_colors))
            rng.shuffle(color_order)
            
            assigned = False
            for c in color_order:
                valid = True
                for ap_idx in pos_to_aps[pos]:
                    a, b, c2, d = aps[ap_idx]
                    vals = []
                    all_assigned = True
                    for p in (a, b, c2, d):
                        if p == pos:
                            vals.append(c)
                        elif colors[p] != -1:
                            vals.append(colors[p])
                        else:
                            all_assigned = False
                            break
                    if all_assigned and len(set(vals)) < 3:
                        valid = False
                        break
                if valid:
                    colors[pos] = c
                    assigned = True
                    break
            
            if not assigned:
                return None
        return colors
    
    best_result = None
    best_num_colors = n + 1
    
    for num_colors in range(1, 8):
        if time.time() - start_time > deadline - 3.0:
            break
        
        for seed in range(300):
            if time.time() - start_time > deadline - 1.0:
                break
            
            result = greedy_assign(num_colors, seed)
            if result is not None:
                used = set(result)
                if len(used) < best_num_colors:
                    best_num_colors = len(used)
                    best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: unique colors (always valid)
    return list(range(n))
