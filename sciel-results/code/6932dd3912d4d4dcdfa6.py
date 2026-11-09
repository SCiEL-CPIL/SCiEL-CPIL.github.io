import numpy as np
import time
import random

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Precompute all non-constant 4-APs
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_array = np.array(ap_list, dtype=np.int32)
    num_aps = len(ap_array)
    
    # For each vertex, which APs does it appear in?
    vertex_aps = [[] for _ in range(n)]
    for i, (a, b, c, d) in enumerate(ap_list):
        vertex_aps[a].append(i)
        vertex_aps[b].append(i)
        vertex_aps[c].append(i)
        vertex_aps[d].append(i)
    
    # Convert to arrays for faster access
    vertex_aps_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                      for lst in vertex_aps]
    
    def compute_bad_mask(colors):
        """Compute boolean mask of APs with < 3 distinct colors."""
        c0 = colors[ap_array[:, 0]]
        c1 = colors[ap_array[:, 1]]
        c2 = colors[ap_array[:, 2]]
        c3 = colors[ap_array[:, 3]]
        
        # Count distinct colors efficiently
        # distinct >= 3 iff NOT (all same) AND NOT (exactly 2 colors)
        # all same: c0==c1==c2==c3
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # exactly 2 colors: partition into 2 non-empty groups
        # This happens when the 4 values take exactly 2 distinct values
        # Cases: (x,x,x,y), (x,x,y,x), (x,x,y,y), (x,y,x,x), (x,y,x,y), (x,y,y,x), (y,x,x,x), (y,x,x,y), (y,x,y,x), (y,y,x,x), (y,y,x,y), (y,y,y,x)
        # Simpler: count how many pairs are equal
        # Actually, let's use: distinct_count = 1 + [c1!=c0] + [c2 not in {c0,c1}] + [c3 not in {c0,c1,c2}]
        
        d1 = (c1 != c0).astype(np.int8)
        d2 = ((c2 != c0) & (c2 != c1)).astype(np.int8)
        d3 = ((c3 != c0) & (c3 != c1) & (c3 != c2)).astype(np.int8)
        
        distinct_count = 1 + d1 + d2 + d3
        return distinct_count < 3
    
    def update_bad_mask_incremental(colors, bad_mask, v, old_color, new_color):
        """Update bad mask when vertex v changes from old_color to new_color."""
        if old_color == new_color:
            return bad_mask, 0
        
        affected = vertex_aps_arr[v]
        if len(affected) == 0:
            return bad_mask, 0
        
        delta = 0
        for ap_idx in affected:
            ap = ap_array[ap_idx]
            
            # Find position of v in AP
            pos = -1
            for i in range(4):
                if ap[i] == v:
                    pos = i
                    break
            
            # Get current colors
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            # New distinct count
            new_colors = [c0, c1, c2, c3]
            new_distinct = len(set(new_colors))
            new_bad = new_distinct < 3
            
            # Old distinct count
            old_colors = [c0, c1, c2, c3]
            old_colors[pos] = old_color
            old_distinct = len(set(old_colors))
            old_bad = old_distinct < 3
            
            if old_bad and not new_bad:
                delta -= 1
            elif not old_bad and new_bad:
                delta += 1
            
            bad_mask[ap_idx] = new_bad
        
        return bad_mask, delta
    
    # Try to find valid solutions with increasing k
    for k in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 5.0:
            break
        
        # Multiple restarts for this k
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Structured initialization: use modular patterns
            # Pattern 1: color[i] = i % k
            # Pattern 2: color[i] = (i * random_prime) % k  
            # Pattern 3: random with bias toward structured patterns
            
            pattern_type = random.randint(0, 3)
            if pattern_type == 0:
                colors = np.array([i % k for i in range(n)], dtype=np.int32)
            elif pattern_type == 1:
                prime = random.choice([2, 3, 5, 7, 11, 13, 17])
                colors = np.array([(i * prime) % k for i in range(n)], dtype=np.int32)
            elif pattern_type == 2:
                # Block pattern: repeat a random permutation
                block = list(range(k))
                random.shuffle(block)
                colors = np.array([block[i % k] for i in range(n)], dtype=np.int32)
            else:
                colors = np.random.randint(0, k, size=n, dtype=np.int32)
            
            bad_mask = compute_bad_mask(colors)
            bad_count = int(np.sum(bad_mask))
            
            if bad_count == 0:
                return colors.tolist()
            
            # Simulated annealing
            T = 5.0
            T_min = 0.01
            cooling_rate = 0.9995
            max_steps = 50000
            step = 0
            
            while bad_count > 0 and step < max_steps:
                if time.time() > deadline - 1.0:
                    break
                
                step += 1
                T = max(T * cooling_rate, T_min)
                
                # Pick a random vertex
                v = random.randint(0, n - 1)
                old_color = colors[v]
                new_color = random.randint(0, k - 1)
                
                if new_color == old_color:
                    continue
                
                # Apply change
                colors[v] = new_color
                bad_mask, delta = update_bad_mask_incremental(colors, bad_mask, v, old_color, new_color)
                
                if delta <= 0:
                    # Accept improvement or neutral
                    bad_count += delta
                    if bad_count == 0:
                        return colors.tolist()
                else:
                    # Worse: accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        bad_count += delta
                    else:
                        # Revert
                        colors[v] = old_color
                        bad_mask, _ = update_bad_mask_incremental(colors, bad_mask, v, new_color, old_color)
        
        # If we didn't find a solution with k colors, try next k
    
    # Fallback: use a valid coloring with many colors
    # color[i] = i ensures all 4-APs have 4 distinct colors
    return list(range(n))
