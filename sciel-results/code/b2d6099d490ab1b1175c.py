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
    
    # Try to find a valid coloring with minimal colors
    # Use backtracking with union-find to detect bicolored cycles
    
    def has_bicolored_cycle(coloring, edge_set):
        """Check if any pair of colors induces a cycle using union-find."""
        from collections import defaultdict
        # Group edges by color pair
        color_edges = defaultdict(list)
        for u, v in edge_set:
            cu, cv = coloring[u], coloring[v]
            if cu == cv:
                continue  # Same color, not a bicolored edge
            pair = (min(cu, cv), max(cu, cv))
            color_edges[pair].append((u, v))
        
        # For each color pair, check if the induced subgraph has a cycle
        for pair, elist in color_edges.items():
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # Cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in elist:
                if not union(u, v):
                    return True  # Cycle found
        
        return False
    
    # Greedy approach: try colors in order, use backtracking if needed
    # First, try to find a valid coloring with k colors for k=2,3,4,...
    
    # For efficiency, use a simple greedy with local search
    # Start with a random assignment and fix violations
    
    import random
    random.seed(42)
    
    # Try increasing number of colors
    for max_colors in range(2, 15):
        # Try multiple random initial colorings
        for attempt in range(100):
            coloring = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Greedy improvement: try to reduce colors
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    old_color = coloring[i]
                    # Try each color from 0 to max_colors-1
                    for new_color in range(max_colors):
                        if new_color == old_color:
                            continue
                        # Check if new_color conflicts with neighbors
                        valid = True
                        for j in adj[i]:
                            if coloring[j] == new_color:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Check if this change creates a bicolored cycle
                        old_val = coloring[i]
                        coloring[i] = new_color
                        if not has_bicolored_cycle(coloring, edges):
                            if new_color < old_color:
                                improved = True
                                break
                            elif new_color == old_color:
                                break
                            else:
                                # Only accept if it doesn't increase max color used
                                max_used = max(coloring)
                                if max_used < old_color or (new_color <= max_used and new_color < old_color):
                                    improved = True
                                    break
                        else:
                            coloring[i] = old_val
                
                # Try to reduce the number of distinct colors used
                used_colors = set(coloring)
                if len(used_colors) < max_colors:
                    # Remap colors to 0..len-1
                    color_map = {}
                    for idx, c in enumerate(sorted(used_colors)):
                        color_map[c] = idx
                    coloring = [color_map[c] for c in coloring]
                    improved = True
            
            # Check if valid
            if not has_bicolored_cycle(coloring, edges):
                # Verify no adjacent same color
                valid = True
                for u, v in edges:
                    if coloring[u] == coloring[v]:
                        valid = False
                        break
                if valid:
                    distinct = len(set(coloring))
                    if distinct <= max_colors:
                        return [int(c) for c in coloring]
    
    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
