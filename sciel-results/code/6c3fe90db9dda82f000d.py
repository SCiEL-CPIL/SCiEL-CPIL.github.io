import time
import itertools

N = 16
DEADLINE = 85  # seconds, safety margin before 90s kill

# Precomputed optimal colors g(n) for n=1..16
G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]

# Precomputed best split k for n=4..16
SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    start_time = time.time()
    
    # Initialize the color matrix
    colors = [[0] * N for _ in range(N)]
    next_color = [0]
    
    def assign_colors(vertices):
        n = len(vertices)
        if n <= 1:
            return next_color[0]
        
        if n == 2:
            u, v = vertices
            c = next_color[0]
            colors[u][v] = c
            colors[v][u] = c
            next_color[0] += 1
            return next_color[0]
        
        if n == 3:
            u, v, w = vertices
            c1 = next_color[0]
            c2 = next_color[0] + 1
            c3 = next_color[0] + 2
            colors[u][v] = c1
            colors[v][u] = c1
            colors[v][w] = c2
            colors[w][v] = c2
            colors[u][w] = c3
            colors[w][u] = c3
            next_color[0] += 3
            return next_color[0]
        
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color[0] += 1
        
        assign_colors(v1)
        assign_colors(v2)
        
        return next_color[0]
    
    all_vertices = list(range(N))
    assign_colors(all_vertices)
    
    # Get current set of used colors
    used_colors = set()
    for i in range(N):
        for j in range(i+1, N):
            used_colors.add(colors[i][j])
    
    best_colors = [colors[i][j] for i in range(N) for j in range(i+1, N)]
    best_num_colors = len(used_colors)
    
    # Precompute all 4-cycles for each edge
    # For edge (u,v), find all 4-cycles containing it
    # A 4-cycle with edge (u,v) involves two other vertices w,x
    # The three possible 4-cycles on {u,v,w,x} that contain edge (u,v):
    # 1. u-v-w-x-u: edges (u,v), (v,w), (w,x), (x,u)
    # 2. u-v-x-w-u: edges (u,v), (v,x), (x,w), (w,u)
    # Note: u-w-v-x-u does NOT contain edge (u,v)
    
    edge_cycles = {}
    edges_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    for (u, v) in edges_list:
        cycles = []
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                # Cycle 1: u-v-w-x-u
                # Edges: (u,v), (v,w), (w,x), (x,u)
                cycles.append(((u, v), (v, w), (w, x), (x, u)))
                # Cycle 2: u-v-x-w-u
                # Edges: (u,v), (v,x), (x,w), (w,u)
                cycles.append(((u, v), (v, x), (x, w), (w, u)))
        edge_cycles[(u, v)] = cycles
    
    def is_valid_change(u, v, new_c):
        """Check if changing edge (u,v) to new_c creates no rainbow C4."""
        for cycle in edge_cycles[(u, v)]:
            e1, e2, e3, e4 = cycle
            c1 = new_c if (e1 == (u, v) or e1 == (v, u)) else colors[e1[0]][e1[1]]
            c2 = colors[e2[0]][e2[1]]
            c3 = colors[e3[0]][e3[1]]
            c4 = colors[e4[0]][e4[1]]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True
    
    # Try to improve
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for (u, v) in edges_list:
            if (time.time() - start_time) > DEADLINE:
                break
            old_c = colors[u][v]
            # Try new colors not currently used
            for new_c in range(120):
                if new_c == old_c or new_c in used_colors:
                    continue
                if is_valid_change(u, v, new_c):
                    # Accept the change
                    colors[u][v] = new_c
                    colors[v][u] = new_c
                    used_colors.add(new_c)
                    best_num_colors = len(used_colors)
                    improved = True
                    break  # Move to next edge
    
    # Second pass: try swapping colors between edges to free up a color for a new one
    if (time.time() - start_time) < DEADLINE - 5:
        for _ in range(3):  # Multiple swap passes
            if (time.time() - start_time) > DEADLINE:
                break
            swapped = False
            for (u1, v1) in edges_list:
                if (time.time() - start_time) > DEADLINE:
                    break
                for (u2, v2) in edges_list:
                    if (time.time() - start_time) > DEADLINE:
                        break
                    if (u1, v1) >= (u2, v2):
                        continue
                    c1 = colors[u1][v1]
                    c2 = colors[u2][v2]
                    if c1 == c2:
                        continue
                    # Check if swapping c1 and c2 is valid
                    # We need to check all cycles involving (u1,v1) and (u2,v2)
                    valid = True
                    for cycle in edge_cycles[(u1, v1)]:
                        e1, e2, e3, e4 = cycle
                        c1_new = c2 if (e1 == (u1, v1) or e1 == (v1, u1)) else colors[e1[0]][e1[1]]
                        c2_new = colors[e2[0]][e2[1]]
                        c3_new = colors[e3[0]][e3[1]]
                        c4_new = colors[e4[0]][e4[1]]
                        if len({c1_new, c2_new, c3_new, c4_new}) == 4:
                            valid = False
                            break
                    if valid:
                        for cycle in edge_cycles[(u2, v2)]:
                            e1, e2, e3, e4 = cycle
                            c1_new = c1 if (e1 == (u2, v2) or e1 == (v2, u2)) else colors[e1[0]][e1[1]]
                            c2_new = colors[e2[0]][e2[1]]
                            c3_new = colors[e3[0]][e3[1]]
                            c4_new = colors[e4[0]][e4[1]]
                            if len({c1_new, c2_new, c3_new, c4_new}) == 4:
                                valid = False
                                break
                    if valid:
                        # Perform swap
                        colors[u1][v1] = c2
                        colors[v1][u1] = c2
                        colors[u2][v2] = c1
                        colors[v2][u2] = c1
                        swapped = True
                        # Try to add a new color after swap
                        for (u3, v3) in edges_list:
                            if (time.time() - start_time) > DEADLINE:
                                break
                            old_c3 = colors[u3][v3]
                            for new_c3 in range(120):
                                if new_c3 == old_c3 or new_c3 in used_colors:
                                    continue
                                if is_valid_change(u3, v3, new_c3):
                                    colors[u3][v3] = new_c3
                                    colors[v3][u3] = new_c3
                                    used_colors.add(new_c3)
                                    best_num_colors = len(used_colors)
                                    break
                            if (time.time() - start_time) > DEADLINE:
                                break
                        if swapped:
                            break
                    if swapped:
                        break
            if not swapped:
                break
    
    # Build result
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(colors[i][j])
    
    return result
