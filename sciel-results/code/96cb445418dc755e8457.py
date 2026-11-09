"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s limit
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Convert to numpy arrays for faster access
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorization."""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        
        # An AP is bad if it uses <= 2 distinct colors
        # Count distinct values: if <= 2, it's bad
        # Vectorized check: 
        # bad if (c0==c1==c2==c3) or 
        #        (exactly 2 distinct: c0==c1 and c2==c3 but c0!=c2, or 
        #         c0==c2 and c1==c3 but c0!=c1, or 
        #         c0==c3 and c1==c2 but c0!=c1, or
        #         c0==c1 and c1==c2 but c2!=c3, etc.)
        # Simpler: compute number of distinct values
        # Use the fact that for 4 elements, <= 2 distinct means:
        # (c0==c1 and c2==c3) OR (c0==c2 and c1==c3) OR (c0==c3 and c1==c2)
        # covers all cases with <= 2 distinct? Let's verify:
        # 4 same: all pairs equal -> covered
        # aabb: c0==c1, c2==c3 -> covered
        # abab: c0==c2, c1==c3 -> covered  
        # abba: c0==c3, c1==c2 -> covered
        # aabc: c0==c1, but c2!=c3, c0!=c2, c0!=c3 -> not covered by above
        #       But aabc has 3 distinct, so it's GOOD
        # aabb has 2 distinct -> bad
        # So the condition (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        # correctly identifies exactly the cases with <= 2 distinct colors
        
        bad = ((c0 == c1) & (c2 == c3)) | \
              ((c0 == c2) & (c1 == c3)) | \
              ((c0 == c3) & (c1 == c2))
        
        return np.sum(bad)
    
    def delta_cost(colors, i, new_color):
        """Calculate change in bad count when changing position i to new_color."""
        old_color = colors[i]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in idx_to_aps_arr[i]:
            ap = aps_arr[ap_idx]
            c = [colors[j] for j in ap]
            pos_in_ap = np.where(ap == i)[0][0]
            
            # Old state
            old_c = c.copy()
            old_bad = 1 if (
                (old_c[0] == old_c[1] and old_c[2] == old_c[3]) or
                (old_c[0] == old_c[2] and old_c[1] == old_c[3]) or
                (old_c[0] == old_c[3] and old_c[1] == old_c[2])
            ) else 0
            
            # New state
            c[pos_in_ap] = new_color
            new_bad = 1 if (
                (c[0] == c[1] and c[2] == c[3]) or
                (c[0] == c[2] and c[1] == c[3]) or
                (c[0] == c[3] and c[1] == c[2])
            ) else 0
            
            delta += new_bad - old_bad
        
        return delta
    
    # Try to find coloring with increasing number of colors
    best_coloring = None
    best_num_colors = 128
    
    # Try different color counts from low to high
    for k in range(3, 10):
        if time.time() > deadline:
            break
        
        # Multiple restarts with local search
        found = False
        max_restarts = 30
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_coloring = colors.tolist()
                best_num_colors = k
                found = True
                break
            
            # Simulated Annealing
            T = 1.0
            no_improve = 0
            max_iters = 5000
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                if current_cost == 0:
                    best_coloring = colors.tolist()
                    best_num_colors = k
                    found = True
                    break
                
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                
                # Calculate delta cost
                dc = delta_cost(colors, i, new_color)
                
                if dc < 0:
                    colors[i] = new_color
                    current_cost += dc
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-dc / T):
                        colors[i] = new_color
                        current_cost += dc
                    else:
                        no_improve += 1
                        if no_improve > 50:
                            break
                
                # Cool down
                T *= 0.999
                if T < 0.01:
                    T = 0.01
            
            if current_cost < best_num_colors:
                # Track best even if not zero
                pass
            
            if found:
                break
        
        if found:
            break
    
    # Fallback: greedy construction if local search didn't find solution
    if best_coloring is None or best_num_colors > 10:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                    colors.append(color)
                    break
        best_coloring = colors
    
    return best_coloring
# EVOLVE-BLOCK-END
