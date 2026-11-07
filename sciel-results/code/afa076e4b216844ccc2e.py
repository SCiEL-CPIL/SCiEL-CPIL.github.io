"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try 3-coloring with pattern (row + 2*col) % 3
    # Map vertex index to (row, col) in 5x10 grid
    # Vertex i: row = i // 10, col = i % 10
    colors = []
    for i in range(n):
        r = i // 10
        c = i % 10
        colors.append((r + 2 * c) % 3)
    
    # Verify: proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            break
    
    # Verify: no bicolored cycles using union-find
    def has_bicolored_cycle(colors, edges, n, c1, c2):
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
                return True  # cycle found
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return False
        
        for u, v in edges:
            if colors[u] == c1 and colors[v] == c2:
                if union(u, v):
                    return True
        return False
    
    valid = True
    for c1 in range(3):
        for c2 in range(c1 + 1, 3):
            if has_bicolored_cycle(colors, edges, n, c1, c2):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return colors
    
    # Fallback: use more colors if needed
    # Try 4-coloring
    colors4 = []
    for i in range(n):
        r = i // 10
        c = i % 10
        colors4.append((r + c) % 4)
    
    valid4 = True
    for u, v in edges:
        if colors4[u] == colors4[v]:
            valid4 = False
            break
    
    if valid4:
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
                if has_bicolored_cycle(colors4, edges, n, c1, c2):
                    valid4 = False
                    break
            if not valid4:
                break
    
    if valid4:
        return colors4
    
    # Last resort: unique colors
    return list(range(n))
# EVOLVE-BLOCK-END

