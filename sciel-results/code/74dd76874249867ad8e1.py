N = 16

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    
    deadline = time.time() + 55.0
    n_edges = N * (N - 1) // 2
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles
    cycles = []
    edge_to_cycles = [[] for _ in range(n_edges)]
    
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        cycles.append(tuple(e1))
        for e in e1:
            edge_to_cycles[e].append(len(cycles)-1)
        
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        cycles.append(tuple(e2))
        for e in e2:
            edge_to_cycles[e].append(len(cycles)-1)
        
        e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        cycles.append(tuple(e3))
        for e in e3:
            edge_to_cycles[e].append(len(cycles)-1)
    
    colors = [0] * n_edges
    color_counts = [0] * 120
    color_counts[0] = n_edges
    distinct_count = 1
    
    def is_valid_edge_change(edge, new_color, cur_colors):
        for cyc in edge_to_cycles[edge]:
            e0, e1, e2, e3 = cycles[cyc]
            c0 = cur_colors[e0] if e0 != edge else new_color
            c1 = cur_colors[e1] if e1 != edge else new_color
            c2 = cur_colors[e2] if e2 != edge else new_color
            c3 = cur_colors[e3] if e3 != edge else new_color
            
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    random.seed(123)
    
    # Local search
    # We prioritize introducing new colors.
    # We maintain color_counts to know which colors are unused.
    
    while time.time() < deadline:
        e = random.randint(0, n_edges - 1)
        old_color = colors[e]
        
        # Try to find a valid new color
        # First, try unused colors (to increase distinct_count)
        # Then, try used colors (to explore)
        
        # To avoid O(120) scan for unused colors every time, 
        # we can just pick random colors and check counts.
        
        # Strategy:
        # 1. Pick a random color nc.
        # 2. If valid, accept.
        # To bias towards new colors, we can try a few random colors, 
        # and if any is unused and valid, prefer it.
        
        best_nc = -1
        best_is_new = False
        
        # Try 5 random candidates
        for _ in range(5):
            nc = random.randint(0, 119)
            if nc == old_color:
                continue
                
            is_new = (color_counts[nc] == 0)
            
            if is_valid_edge_change(e, nc, colors):
                # If this is a new color, it's "better"
                # But we just want to maximize count. 
                # Any new color increases count by 1.
                # Any old color keeps count same (or decreases if it was the last one).
                
                # Let's just accept the first valid one?
                # Or prefer new?
                
                # If we accept the first valid one, we might get stuck 
                # using only 2 colors if they are always valid.
                
                # So, we should prefer new colors.
                
                if is_new:
                    best_nc = nc
                    best_is_new = True
                    break # Found a new color, take it immediately
                
                if best_nc == -1:
                    best_nc = nc
                    best_is_new = False
        
        if best_nc != -1:
            # Apply change
            colors[e] = best_nc
            
            # Update counts
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                distinct_count -= 1
                
            color_counts[best_nc] += 1
            if color_counts[best_nc] == 1:
                distinct_count += 1

    return colors
# EVOLVE-BLOCK-END
