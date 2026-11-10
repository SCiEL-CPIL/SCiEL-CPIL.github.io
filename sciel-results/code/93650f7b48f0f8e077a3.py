"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def is_valid_coloring(colors, max_color):
        """Check if coloring is valid: adjacent differ, and each pair of colors induces a forest."""
        # Check adjacent vertices have different colors
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Check each pair of colors induces a forest using union-find
        for c1 in range(max_color + 1):
            for c2 in range(c1 + 1, max_color + 1):
                # Union-find for vertices with color c1 or c2
                parent = list(range(n))
                rank = [0] * n
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                for u, v in edges:
                    cu, cv = colors[u], colors[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        if not union(u, v):
                            return False  # Bicolored cycle found
        
        return True
    
    # Try to find minimum number of colors
    # Start with 3 colors (2 won't work due to cycles)
    best_colors = list(range(n))  # fallback: all different
    best_num = n
    
    deadline = time.time() + 50  # safety margin
    
    for num_colors in range(3, 10):
        if time.time() > deadline:
            break
        if num_colors >= best_num:
            break
        
        # Try to find a valid coloring with num_colors colors
        # Use backtracking with ordering
        colors = [-1] * n
        color_counts = [0] * num_colors
        
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        # Map: position in order -> vertex
        pos_to_vertex = order
        vertex_to_pos = [0] * n
        for i, v in enumerate(order):
            vertex_to_pos[v] = i
        
        found = [False]
        
        def backtrack(pos):
            if time.time() > deadline:
                return
            if found[0]:
                return
            if pos == n:
                found[0] = True
                return
            
            v = pos_to_vertex[pos]
            
            # Try each color
            for c in range(num_colors):
                # Check if color c is available (no neighbor has color c)
                valid = True
                for nb in adj[v]:
                    if vertex_to_pos[nb] < pos and colors[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[v] = c
                color_counts[c] += 1
                
                # Check if adding this color creates a bicolored cycle
                # Only need to check pairs involving color c
                ok = True
                for c2 in range(num_colors):
                    if c2 == c:
                        continue
                    # Check if edges between color c and c2 form a forest
                    parent = list(range(n))
                    rank = [0] * n
                    
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    
                    def union(x, y):
                        px, py = find(x), find(y)
                        if px == py:
                            return False
                        if rank[px] < rank[py]:
                            px, py = py, px
                        parent[py] = px
                        if rank[px] == rank[py]:
                            rank[px] += 1
                        return True
                    
                    for u2, v2 in edges:
                        cu2, cv2 = colors[u2], colors[v2]
                        if cu2 == -1 or cv2 == -1:
                            continue
                        if (cu2 == c and cv2 == c2) or (cu2 == c2 and cv2 == c):
                            if not union(u2, v2):
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    backtrack(pos + 1)
                    if found[0]:
                        return
                
                colors[v] = -1
                color_counts[c] -= 1
        
        backtrack(0)
        
        if found[0]:
            best_colors = colors[:]
            best_num = num_colors
            break
    
    return best_colors
# EVOLVE-BLOCK-END
