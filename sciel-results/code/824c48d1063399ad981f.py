import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    # other_indices[ap_idx][pos_idx] = tuple of other 3 indices
    other_indices = []
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)
    
    # pos_in_ap[ap_idx][idx] = position of idx in ap
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx

    def count_bad_colors(colors):
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5, ...
    for k in range(3, 15):
        if time.time() > deadline - 15:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        restart_time_limit = 0.8
        
        while time.time() < deadline - 10:
            restart_start = time.time()
            
            # Random initialization
            colors = np.random.randint(0, k, n)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing
            T = 1.0
            while time.time() - restart_start < restart_time_limit:
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    pos = pos_in_ap[ap_idx][i]
                    others = other_indices[ap_idx][pos]
                    c_others = [colors[j] for j in others]
                    set_others = set(c_others)
                    size_others = len(set_others)
                    
                    old_size = size_others + (0 if old_color in set_others else 1)
                    new_size = size_others + (0 if new_color in set_others else 1)
                    
                    old_bad = 1 if old_size <= 2 else 0
                    new_bad = 1 if new_size <= 2 else 0
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                        
                T *= 0.998
                if T < 0.01:
                    T = 0.01

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            result = list(best_k_solution)
            used_colors = set(result)
            
            # Try to reduce colors
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                improved = False
                for color in list(used_colors):
                    for target_color in used_colors:
                        if color == target_color:
                            continue
                        test_colors = [target_color if c == color else c for c in result]
                        if count_bad_colors(np.array(test_colors)) == 0:
                            result = test_colors
                            used_colors = set(result)
                            improved = True
                            break
                    if improved:
                        break
                    if time.time() > deadline - 5:
                        break
                if not improved:
                    break
            
            return result
            
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
