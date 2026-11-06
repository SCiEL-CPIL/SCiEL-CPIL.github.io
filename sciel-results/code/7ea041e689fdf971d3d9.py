import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs: (i, i+d, i+2d, i+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # For each position, store which APs it participates in
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count number of APs with < 3 distinct colors."""
        v = 0
        for ap in ap_list:
            vals = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
            if len(set(vals)) < 3:
                v += 1
        return v
    
    def check_valid(colors):
        """Check if coloring is valid (0 violations)."""
        for ap in ap_list:
            vals = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
            if len(set(vals)) < 3:
                return False
        return True
    
    # Try local search with k colors
    def local_search(k, max_iters=500000):
        if time.time() > deadline:
            return None
        
        # Initialize randomly
        colors = np.random.randint(0, k, size=n)
        cur_violations = count_violations(colors)
        best_colors = colors.copy()
        best_violations = cur_violations
        
        if best_violations == 0:
            return best_colors.tolist()
        
        # Simulated annealing
        temp = 10.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random position and change its color
            pos = np.random.randint(0, n)
            old_color = colors[pos]
            new_color = np.random.randint(0, k)
            
            if new_color == old_color:
                continue
            
            # Compute delta in violations
            # Only APs containing pos can change violation status
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                ap = ap_list[ap_idx]
                # Before change
                old_vals = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                old_bad = 1 if len(set(old_vals)) < 3 else 0
                
                # After change
                new_vals = old_vals.copy()
                for i, p in enumerate(ap):
                    if p == pos:
                        new_vals[i] = new_color
                new_bad = 1 if len(set(new_vals)) < 3 else 0
                
                delta += new_bad - old_bad
            
            colors[pos] = new_color
            
            new_violations = cur_violations + delta
            
            if new_violations < cur_violations:
                cur_violations = new_violations
                if cur_violations < best_violations:
                    best_violations = cur_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return best_colors.tolist()
            else:
                # Accept with probability exp(-delta/temp)
                if delta > 0:
                    prob = np.exp(-delta / temp)
                    if np.random.random() < prob:
                        cur_violations = new_violations
                    else:
                        colors[pos] = old_color
            
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
        
        if best_violations == 0:
            return best_colors.tolist()
        return None
    
    # Try k=3 first
    for attempt in range(5):
        if time.time() > deadline:
            break
        result = local_search(3)
        if result is not None:
            return result
    
    # Try k=4
    for attempt in range(5):
        if time.time() > deadline:
            break
        result = local_search(4)
        if result is not None:
            return result
    
    # Try k=5
    for attempt in range(3):
        if time.time() > deadline:
            break
        result = local_search(5)
        if result is not None:
            return result
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, n + 1):
        pos = x - 1  # 0-indexed
        best_color = None
        for color in range(128):
            valid = True
            # Check all APs where pos is the last element
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                if a < 0:
                    continue
                vals = [colors[a], colors[a + d], colors[a + 2 * d], color]
                if len(set(vals)) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check APs where pos is not the last element but is included
            # pos = a + d: a = pos - d, need a+2d, a+3d < n and assigned
            # pos = a + 2d: a = pos - 2d, need a+d, a+3d < n and assigned  
            # pos = a: a = pos, need a+d, a+2d, a+3d < n and assigned
            for d in range(1, n):
                # pos as first element: a=pos, a+d, a+2d, a+3d
                if pos + 3 * d < n:
                    if colors[pos + d] >= 0 and colors[pos + 2 * d] >= 0 and colors[pos + 3 * d] >= 0:
                        vals = [color, colors[pos + d], colors[pos + 2 * d], colors[pos + 3 * d]]
                        if len(set(vals)) < 3:
                            valid = False
                            break
                # pos as second element: a=pos-d, a+d=pos, a+2d, a+3d
                if pos - d >= 0 and pos + 2 * d < n:
                    if colors[pos - d] >= 0 and colors[pos + d] >= 0 and colors[pos + 2 * d] >= 0:
                        vals = [colors[pos - d], color, colors[pos + d], colors[pos + 2 * d]]
                        if len(set(vals)) < 3:
                            valid = False
                            break
                # pos as third element: a=pos-2d, a+d, a+2d=pos, a+3d
                if pos - 2 * d >= 0 and pos + d < n:
                    if colors[pos - 2 * d] >= 0 and colors[pos - d] >= 0 and colors[pos + d] >= 0:
                        vals = [colors[pos - 2 * d], colors[pos - d], color, colors[pos + d]]
                        if len(set(vals)) < 3:
                            valid = False
                            break
            if valid:
                best_color = color
                break
        
        if best_color is None:
            # Shouldn't happen with 128 colors available
            best_color = 0
        
        colors.append(best_color)
    
    return colors
