import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Try multiple modular arithmetic constructions and pick the best
    best_colors = None
    best_num_colors = 0
    
    # List of (a, b, mod) tuples for c(i,j) = (a*i + b*j) mod m
    candidates = [
        (1, 0, 19),   # i mod 19
        (0, 1, 19),   # j mod 19
        (1, 1, 19),   # (i+j) mod 19
        (1, 2, 19),   # (i+2j) mod 19
        (2, 1, 19),   # (2i+j) mod 19
        (1, 3, 19),   # (i+3j) mod 19
        (3, 1, 19),   # (3i+j) mod 19
        (1, 4, 19),   # (i+4j) mod 19
        (4, 1, 19),   # (4i+j) mod 19
        (1, 5, 19),   # (i+5j) mod 19
        (5, 1, 19),   # (5i+j) mod 19
        (1, 1, 17),   # (i+j) mod 17
        (1, 2, 17),   # (i+2j) mod 17
        (2, 1, 17),   # (2i+j) mod 17
        (1, 3, 17),   # (i+3j) mod 17
        (3, 1, 17),   # (3i+j) mod 17
        (1, 1, 13),   # (i+j) mod 13
        (1, 2, 13),   # (i+2j) mod 13
        (2, 1, 13),   # (2i+j) mod 13
        (1, 3, 13),   # (i+3j) mod 13
        (3, 1, 13),   # (3i+j) mod 13
        (1, 1, 11),   # (i+j) mod 11
        (1, 2, 11),   # (i+2j) mod 11
        (2, 1, 11),   # (2i+j) mod 11
        (1, 3, 11),   # (i+3j) mod 11
        (3, 1, 11),   # (3i+j) mod 11
        (1, 1, 7),    # (i+j) mod 7
        (1, 2, 7),    # (i+2j) mod 7
        (2, 1, 7),    # (2i+j) mod 7
        (1, 3, 7),    # (i+3j) mod 7
        (3, 1, 7),    # (3i+j) mod 7
    ]
    
    for a, b, m in candidates:
        if time.time() > deadline - 5:
            break
        
        colors = [0] * (N * (N - 1) // 2)
        for i in range(N):
            for j in range(i + 1, N):
                colors[get_idx(i, j)] = (a * i + b * j) % m
        
        # Check validity
        valid = True
        ec = [[-1] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[get_idx(i, j)]
                ec[i][j] = c
                ec[j][i] = c
        
        for i in range(N):
            if not valid:
                break
            for j in range(i + 1, N):
                if not valid:
                    break
                c1 = ec[i][j]
                for x in range(N):
                    if x == i or x == j:
                        continue
                    c2 = ec[i][x]
                    for y in range(N):
                        if y == i or y == j or y == x:
                            continue
                        c3 = ec[x][y]
                        c4 = ec[y][j]
                        if len({c1, c2, c3, c4}) == 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            num_colors = len(set(colors))
            if num_colors > best_num_colors:
                best_num_colors = num_colors
                best_colors = colors.copy()
    
    # If no valid construction found, use a simple valid coloring
    if best_colors is None:
        best_colors = [0] * (N * (N - 1) // 2)
        for i in range(N):
            for j in range(i + 1, N):
                best_colors[get_idx(i, j)] = i % 7
        best_num_colors = len(set(best_colors))
    
    # Local search to add more colors
    colors = best_colors
    used_colors = set(colors)
    all_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.append((i, j))
    
    random.seed(42)
    
    # Build edge color lookup for fast checking
    ec = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c = colors[get_idx(i, j)]
            ec[i][j] = c
            ec[j][i] = c
    
    attempts = 0
    max_attempts = 300000
    
    while time.time() < deadline - 3 and attempts < max_attempts:
        attempts += 1
        
        # Pick a random edge
        edge_idx = random.randint(0, len(all_edges) - 1)
        u, v = all_edges[edge_idx]
        old_color = colors[get_idx(u, v)]
        
        # Try to change to a new color
        unused = [c for c in range(120) if c not in used_colors]
        if not unused:
            break
        
        new_color = random.choice(unused)
        
        # Check if changing this edge to new_color keeps validity
        # We need to check all 4-cycles containing this edge
        valid = True
        for x in range(N):
            if x == u or x == v:
                continue
            cx = ec[u][x]
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                cy = ec[x][y]
                cz = ec[y][v]
                if len({new_color, cx, cy, cz}) == 4:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            colors[get_idx(u, v)] = new_color
            ec[u][v] = new_color
            ec[v][u] = new_color
            used_colors.add(new_color)
    
    # Final verification
    ec_final = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c = colors[get_idx(i, j)]
            ec_final[i][j] = c
            ec_final[j][i] = c
    
    valid_final = True
    for i in range(N):
        if not valid_final:
            break
        for j in range(i + 1, N):
            if not valid_final:
                break
            c1 = ec_final[i][j]
            for x in range(N):
                if x == i or x == j:
                    continue
                c2 = ec_final[i][x]
                for y in range(N):
                    if y == i or y == j or y == x:
                        continue
                    c3 = ec_final[x][y]
                    c4 = ec_final[y][j]
                    if len({c1, c2, c3, c4}) == 4:
                        valid_final = False
                        break
                if not valid_final:
                    break
            if not valid_final:
                break
        if not valid_final:
            break
    
    if not valid_final:
        # Fallback to the best valid construction
        colors = best_colors
    
    return colors
# EVOLVE-BLOCK-END
