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
    
    deadline = time.time() + 55  # 55s safety margin
    
    # Vertex ordering: use degree-based heuristic (all degree 4, so use a good static order)
    # Let's use a simple ordering that helps: order by index but we can try different orderings
    # For now, natural order
    order = list(range(n))
    
    def check_bicolor_forest(color_of, k_colors):
        """Check if for every pair of colors, the induced subgraph is a forest."""
        # For each pair (a,b), check if the subgraph induced by vertices with color a or b has a cycle
        # Use union-find for each pair
        for a in range(k_colors):
            for b in range(a+1, k_colors):
                # Collect edges where both endpoints have color a or b
                # and at least one has the other color (i.e., edges between a and b, or within a, or within b)
                # Actually, the induced subgraph on vertices colored a or b includes ALL edges among those vertices
                # We need to check if it has a cycle
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False  # cycle
                    parent[px] = py
                    return True
                has_cycle = False
                for u, v in edges:
                    cu = color_of[u]
                    cv = color_of[v]
                    # Both must be in {a, b}
                    if cu in (a, b) and cv in (a, b):
                        if not union(u, v):
                            has_cycle = True
                            break
                if has_cycle:
                    return False
        return True
    
    def backtrack(idx, color_of, k):
        """Try to assign colors 0..k-1 to vertices in order[idx:]."""
        if time.time() > deadline:
            return None
        if idx == n:
            return list(color_of)
        
        v = order[idx]
        # Determine forbidden colors: neighbors already colored
        forbidden = set()
        for nb in adj[v]:
            if color_of[nb] != -1:
                forbidden.add(color_of[nb])
        
        for c in range(k):
            if c in forbidden:
                continue
            # Assign
            color_of[v] = c
            # Incremental check: for each pair (c, d) where d is a color used by a neighbor,
            # check if adding v creates a bicolored cycle
            valid = True
            for nb in adj[v]:
                if color_of[nb] != -1 and color_of[nb] != c:
                    d = color_of[nb]
                    # Check if the subgraph induced by colors {c, d} has a cycle
                    # Only need to check edges involving v and the existing {c,d} subgraph
                    # Quick check: does v create a cycle in the {c,d} induced subgraph?
                    # v is color c. Its neighbors in color d are those nb with color d.
                    # Also neighbors in color c.
                    # The {c,d} subgraph: check if adding v (and its edges to c and d neighbors) creates a cycle
                    # Use union-find on the {c,d} induced subgraph
                    parent = list(range(n))
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        px, py = find(x), find(y)
                        if px == py:
                            return False
                        parent[px] = py
                        return True
                    cycle = False
                    for u2, v2 in edges:
                        cu2 = color_of[u2]
                        cv2 = color_of[v2]
                        if cu2 in (c, d) and cv2 in (c, d):
                            if not union(u2, v2):
                                cycle = True
                                break
                    if cycle:
                        valid = False
                        break
            
            if valid:
                result = backtrack(idx + 1, color_of, k)
                if result is not None:
                    return result
            color_of[v] = -1
        return None
    
    # Try increasing k
    for k in range(2, 10):
        color_of = [-1] * n
        result = backtrack(0, color_of, k)
        if result is not None:
            return [int(x) for x in result]
    
    # Fallback: should not happen
    return list(range(n))
# EVOLVE-BLOCK-END
