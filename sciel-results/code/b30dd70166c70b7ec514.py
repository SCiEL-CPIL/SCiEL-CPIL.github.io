"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
N = 16

# EVOLVE-BLOCK-START
def construct():
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)
        
    edge_list = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_list.append((i, j))
            
    edge_index = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    colors = [0] * len(edge_list)
    for i in range(N):
        for j in range(i + 1, N):
            colors[get_idx(i, j)] = i
            
    c4_partners = [[] for _ in range(len(edge_list))]
    for idx, (u, v) in enumerate(edge_list):
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(w + 1, N):
                if x == u or x == v:
                    continue
                e_vx = edge_index[(v, x)]
                e_xw = edge_index[(x, w)]
                e_wu = edge_index[(w, u)]
                c4_partners[idx].append((e_vx, e_xw, e_wu))
                
    color_counts = [0] * 120
    for c in colors:
        color_counts[c] += 1
    used_colors = set(c for c in range(120) if color_counts[c] > 0)
    
    # Local Search
    # Increase iterations to ensure convergence
    for _ in range(200000):
        idx = random.randint(0, len(edge_list) - 1)
        old_color = colors[idx]
        
        # Bias towards new colors
        if random.random() < 0.7 and len(used_colors) < 120:
            # Pick a random unused color
            # To avoid O(120) scan every time, we can just pick random 0..119
            # and check if it's unused?
            # Or maintain a list of unused colors?
            # Simple: pick random 0..119. If it's unused, great. If not, it's a shuffle.
            # But we want to FORCE new colors.
            # Let's just pick a random color from 0..119.
            # The probability of hitting an unused color is (120 - used)/120.
            # As used increases, this drops.
            # To be more effective, we can explicitly pick from unused if we keep a list.
            # But rebuilding the list of unused is O(120).
            # Let's just pick random. The random walk will eventually explore.
            
            new_color = random.randint(0, 119)
        else:
            new_color = random.randint(0, 119)
            
        if new_color == old_color:
            continue
            
        valid = True
        for e_vx, e_xw, e_wu in c4_partners[idx]:
            c1 = new_color
            c2 = colors[e_vx]
            c3 = colors[e_xw]
            c4 = colors[e_wu]
            # Optimization: check distinctness without set creation if possible?
            # Set creation is fine for 4 elements.
            if len({c1, c2, c3, c4}) == 4:
                valid = False
                break
        
        if valid:
            # Update state
            colors[idx] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            if color_counts[old_color] == 0:
                used_colors.remove(old_color)
            if color_counts[new_color] == 1:
                used_colors.add(new_color)
                
    return colors
# EVOLVE-BLOCK-END
