"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def has_bicolored_cycle(coloring, c1, c2):
        """Check if the subgraph induced by colors c1 and c2 has a cycle using union-find."""
        # Collect vertices with color c1 or c2
        vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
        if len(vertices) < 3:
            return False
        
        # Union-find
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False  # cycle detected
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Check all edges within the bicolored subgraph
        for u in vertices:
            for v in adj[u]:
                if v > u and coloring[v] in (c1, c2):
                    if not union(u, v):
                        return True
        return False
    
    def is_valid(coloring):
        """Check if coloring is valid: proper and no bicolored cycles."""
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored cycles
        colors_used = set(coloring)
        for c1, c2 in itertools.combinations(colors_used, 2):
            if has_bicolored_cycle(coloring, c1, c2):
                return False
        return True
    
    def is_valid_partial(coloring, vertex):
        """Check if assigning coloring[vertex] is consistent so far."""
        c = coloring[vertex]
        # Check neighbors have different colors
        for nb in adj[vertex]:
            if coloring[nb] == c:
                return False
        
        # Check bicolored cycles involving this vertex's color
        colors_used = set(coloring[i] for i in range(vertex+1) if coloring[i] is not None)
        if c not in colors_used:
            return True
            
        for other_c in colors_used:
            if other_c == c:
                continue
            # Check if adding this vertex creates a bicolored cycle
            # We need to check if the bicolored subgraph (c, other_c) has a cycle
            # that includes this vertex
            # Quick check: if vertex has two neighbors with color other_c that are 
            # connected in the bicolored subgraph, that would form a cycle
            # More precisely, use union-find on the partial bicolored subgraph
            vertices = [i for i in range(vertex+1) if coloring[i] in (c, other_c)]
            if len(vertices) < 3:
                continue
                
            parent = {v: v for v in vertices}
            rank = {v: 0 for v in vertices}
            
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
            
            for u in vertices:
                for v in adj[u]:
                    if v > u and v <= vertex and coloring[v] in (c, other_c):
                        if not union(u, v):
                            return False
        return True
    
    # Try increasing number of colors
    for num_colors in range(2, 10):
        if time.time() > deadline:
            break
            
        # Use greedy + backtracking
        best_coloring = None
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # Try different orderings to find a solution
        def backtrack(idx, coloring, colors_used):
            nonlocal best_coloring
            if time.time() > deadline:
                return None
            
            if idx == len(order):
                # Full coloring found
                return coloring[:]
            
            v = order[idx]
            
            # Try each color
            for c in range(num_colors):
                # Check if color c is allowed
                allowed = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        allowed = False
                        break
                if not allowed:
                    continue
                
                coloring[v] = c
                if is_valid_partial(coloring, v):
                    result = backtrack(idx + 1, coloring, colors_used)
                    if result is not None:
                        return result
                coloring[v] = None
            
            return None
        
        # Try with vertex ordering
        coloring = [None] * n
        result = backtrack(0, coloring, set())
        if result is not None:
            if is_valid(result):
                best_coloring = result
                break
    
    if best_coloring is None:
        # Fallback: use all distinct colors
        best_coloring = list(range(n))
    
    # Ensure all values are Python ints
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
