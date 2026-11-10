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
    
    def check_bicolored_cycle(coloring):
        """Check if any pair of colors induces a cycle using union-find."""
        from collections import defaultdict
        color_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu == cv:
                return False  # Adjacent same color is invalid
            pair = (min(cu, cv), max(cu, cv))
            color_edges[pair].append((u, v))
        
        for pair, elist in color_edges.items():
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            for u, v in elist:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return True  # Cycle found
                parent[ru] = rv
        
        return False
    
    # Use constraint-based backtracking with forward checking
    # Order vertices by degree (highest first) for better pruning
    vertices_by_degree = sorted(range(n), key=lambda x: -len(adj[x]))
    
    def backtrack(idx, coloring, used_colors):
        if idx == n:
            return coloring.copy()
        
        v = vertices_by_degree[idx]
        
        # Try each color from 0 to max(used_colors)
        max_color_to_try = max(used_colors) if used_colors else 0
        for c in range(max_color_to_try + 1):
            # Check neighbor constraint
            valid = True
            for nb in adj[v]:
                if coloring[nb] is not None and coloring[nb] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Tentatively assign
            coloring[v] = c
            
            # Forward checking: verify no bicolored cycle created
            if check_bicolored_cycle(coloring):
                coloring[v] = None
                continue
            
            # Recurse
            result = backtrack(idx + 1, coloring, used_colors | {c})
            if result is not None:
                return result
            
            coloring[v] = None
        
        return None
    
    # Start with empty coloring
    coloring = [None] * n
    result = backtrack(0, coloring, set())
    
    if result is not None:
        # Remap colors to 0..k-1
        color_map = {}
        for c in sorted(set(result)):
            color_map[c] = len(color_map)
        result = [color_map[c] for c in result]
        return [int(c) for c in result]
    
    # Fallback: greedy with local search
    import random
    random.seed(42)
    
    for max_colors in range(3, 15):
        for attempt in range(200):
            coloring = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    for new_color in range(max_colors):
                        if new_color == coloring[i]:
                            continue
                        # Check neighbors
                        valid = True
                        for j in adj[i]:
                            if coloring[j] == new_color:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        old = coloring[i]
                        coloring[i] = new_color
                        if not check_bicolored_cycle(coloring):
                            if new_color < old or (max(coloring) < old):
                                improved = True
                                break
                        else:
                            coloring[i] = old
            
            if not check_bicolored_cycle(coloring):
                valid = True
                for u, v in edges:
                    if coloring[u] == coloring[v]:
                        valid = False
                        break
                if valid:
                    distinct = len(set(coloring))
                    if distinct <= max_colors:
                        color_map = {}
                        for c in sorted(set(coloring)):
                            color_map[c] = len(color_map)
                        coloring = [color_map[c] for c in coloring]
                        return [int(c) for c in coloring]
    
    return list(range(n))
# EVOLVE-BLOCK-END
