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
    
    def check_bicolored_forest(coloring, num_colors):
        """Check that for every pair of colors, the induced subgraph is a forest."""
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                # Get vertices with color c1 or c2
                vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
                if len(vertices) < 3:
                    continue
                # Use union-find to detect cycles
                parent = {v: v for v in vertices}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    parent[rx] = ry
                    return True
                
                has_cycle = False
                for v in vertices:
                    for nb in adj[v]:
                        if nb > v and coloring[nb] in (c1, c2):
                            if not union(v, nb):
                                has_cycle = True
                                break
                    if has_cycle:
                        break
                if has_cycle:
                    return False
        return True
    
    def is_valid(coloring, num_colors):
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        return check_bicolored_forest(coloring, num_colors)
    
    # Try 3 colors with backtracking
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    def backtrack(idx, coloring):
        if idx == len(order):
            return is_valid(coloring, 3)
        
        v = order[idx]
        for c in range(3):
            # Check no neighbor has color c
            ok = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            coloring[v] = c
            if backtrack(idx + 1, coloring):
                return True
            coloring[v] = -1
        
        return False
    
    coloring = [-1] * n
    if backtrack(0, coloring):
        return [int(c) for c in coloring]
    
    # Fallback to 4 colors
    coloring = [i % 4 for i in range(n)]
    # Fix conflicts
    for u, v in edges:
        if coloring[u] == coloring[v]:
            used = set(coloring[nb] for nb in adj[v])
            for c in range(4):
                if c not in used:
                    coloring[v] = c
                    break
    
    if is_valid(coloring, 4):
        return [int(c) for c in coloring]
    
    # Last resort: 5 colors
    coloring = [i % 5 for i in range(n)]
    for u, v in edges:
        if coloring[u] == coloring[v]:
            used = set(coloring[nb] for nb in adj[v])
            for c in range(5):
                if c not in used:
                    coloring[v] = c
                    break
    
    return [int(c) for c in coloring]
