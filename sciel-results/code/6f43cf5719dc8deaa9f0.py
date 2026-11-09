import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    A = np.array(aps, dtype=np.int32)
    
    idx_to_aps = [[] for _ in range(n)]
    other_indices = []
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)
    
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx
    
    def count_bad_colors_vectorized(colors):
        c = colors[A]
        sorted_c = np.sort(c, axis=1)
        diffs = np.diff(sorted_c, axis=1)
        zero_counts = (diffs == 0).sum(axis=1)
        return np.sum(zero_counts >= 2)
    
    def check_valid(colors):
        return count_bad_colors_vectorized(colors) == 0
    
    # Precompute for each AP the 4 positions
    ap_positions = A  # shape (num_aps, 4)
    
    # Strategy: Try periodic patterns first
    # For a periodic pattern of period p with k colors, check if it's valid
    def check_periodic(pattern, p, k):
        if time.time() > deadline - 5:
            return False
        colors = np.array([pattern[i % p] for i in range(n)], dtype=np.int32)
        return check_valid(colors)
    
    # Try small periods with k=3
    for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        if time.time() > deadline - 30:
            break
        
        k = 3
        total = k ** p
        if total > 50000:
            continue
            
        for pattern_tuple in product(range(k), repeat=p):
            if time.time() > deadline - 25:
                break
            if check_periodic(pattern_tuple, p, k):
                colors = np.array([pattern_tuple[i % p] for i in range(n)], dtype=np.int32)
                return list(colors)
    
    # Strategy 2: Local search
    best_solution = None
    best_cost = float('inf')
    
    for k in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Use a smarter initialization: start with random and do SA
        num_restarts = 300
        steps_per_restart = 200
        
        for restart in range(num_restarts):
            if time.time() > deadline - 3:
                break
            
            colors = np.random.randint(0, k, n, dtype=np.int32)
            current_cost = count_bad_colors_vectorized(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
            
            T = 1.0
            for step in range(steps_per_restart):
                if time.time() > deadline - 2:
                    break
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
                    c0, c1, c2 = colors[others[0]], colors[others[1]], colors[others[2]]
                    
                    # Count distinct in {old_color, c0, c1, c2}
                    # Bad if distinct <= 2
                    # Equivalent to: old_color appears >= 3 times OR some other color appears >= 3 times
                    # Faster: check if all 4 are in a set of size <= 2
                    if c0 == c1 and c1 == c2:
                        old_bad = 1 if old_color == c0 else 0
                        new_bad = 1 if new_color == c0 else 0
                    elif c0 == c1:
                        old_bad = 1 if old_color == c0 or old_color == c2 else 0
                        new_bad = 1 if new_color == c0 or new_color == c2 else 0
                    elif c0 == c2:
                        old_bad = 1 if old_color == c0 or old_color == c1 else 0
                        new_bad = 1 if new_color == c0 or new_color == c1 else 0
                    elif c1 == c2:
                        old_bad = 1 if old_color == c1 or old_color == c0 else 0
                        new_bad = 1 if new_color == c1 or new_color == c0 else 0
                    else:
                        old_bad = 1 if old_color in (c0, c1, c2) else 0
                        new_bad = 1 if new_color in (c0, c1, c2) else 0
                    
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                        
                T *= 0.998
                if T < 0.05:
                    T = 0.05
            
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break
        
        if best_k_cost == 0:
            if check_valid(best_k_solution):
                return list(best_k_solution)
        
        if best_k_solution is not None and best_k_cost < best_cost:
            best_cost = best_k_cost
            best_solution = best_k_solution
    
    if best_solution is not None:
        return list(best_solution)
    
    # Greedy fallback
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3+1):
                if x-1-3*d >= 0:
                    ap_colors = [color, colors[x-1-d], colors[x-1-2*d], colors[x-1-3*d]]
                    if len(set(ap_colors)) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    return colors
