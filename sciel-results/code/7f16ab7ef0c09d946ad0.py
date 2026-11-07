"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Vertex ordering: highest degree first
    vertex_order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Map original index to position in order? 
    # It's easier to just iterate through vertex_order and assign to coloring[original_index]
    
    def is_valid_assignment(coloring, v, c):
        # Check proper coloring
        for u in adj[v]:
            if coloring[u] == c:
                return False
        
        # Check bicolored forest constraint
        # For each other color d, check if adding v (color c) creates a cycle in {c, d} subgraph
        # A cycle is created if v has two neighbors with color d (or c) that are connected in the current {c, d} subgraph.
        # Actually, v has color c. Neighbors with color c are not allowed (proper coloring).
        # So we only care about neighbors with color d != c.
        # Let N_d be the set of neighbors of v with color d.
        # If |N_d| >= 2, check if any two are connected in the subgraph induced by colors {c, d}.
        
        for d in range(len(set(coloring))): # Iterate over used colors? Or all possible colors?
             # We only need to check d that are actually used in the coloring so far, or potentially will be?
             # The constraint is on the final coloring. But we can prune early.
             # If we haven't used color d yet, adding v with color c doesn't create a cycle with d 
             # because there are no vertices with color d (except potentially future ones, but we can't check future).
             # Wait, if d is not used, the subgraph {c, d} only has vertices with color c.
             # A subgraph with only one color is a forest (no edges, or edges only if self-loops which don't exist).
             # So we only need to check d that are currently present in the coloring.
             pass
        
        used_colors = set(coloring)
        if c not in used_colors:
            # If c is new, no other vertex has color c.
            # So for any d, the {c, d} subgraph has v (color c) and some vertices with color d.
            # v is connected to neighbors with color d.
            # Since v is the only vertex with color c, it cannot be part of a cycle 
            # because a cycle requires at least two vertices of the same color to close the loop?
            # No, a cycle in {c, d} can alternate c, d, c, d...
            # If v is the only c, the cycle must go v -> d1 -> ... -> d2 -> v.
            # This requires d1 and d2 to be connected in the d-subgraph?
            # No, the path d1 -> ... -> d2 must use only vertices with color c or d.
            # Since v is the only c, the path cannot use any other c.
            # So the path must be entirely in d? But d-subgraph has no edges between d's? 
            # Wait, edges exist between d and d? No, proper coloring forbids d-d edges.
            # So edges are only c-d.
            # So a cycle must be v(c) - u1(d) - u2(c) ...
            # If v is the only c, there is no u2(c).
            # So no cycle can involve v if v is the unique vertex of color c.
            # Thus, if c is new, it is always safe w.r.t. bicolored cycles?
            # Let's verify.
            # Cycle: v(c) - a(d) - b(c) - c(d) - v(c).
            # Requires two c's. If v is the only c, impossible.
            # So yes, assigning a new color is always safe for the bicolored constraint.
            return True
        else:
            # c is already used.
            for d in used_colors:
                if d == c:
                    continue
                # Find neighbors of v with color d
                neigh_d = [u for u in adj[v] if coloring[u] == d]
                if len(neigh_d) < 2:
                    continue
                
                # Check if any pair in neigh_d is connected in the {c, d} subgraph
                # We can do a BFS from neigh_d[0] to see if it reaches any other in neigh_d
                # using only vertices with color c or d.
                
                visited = set()
                queue = [neigh_d[0]]
                visited.add(neigh_d[0])
                
                while queue:
                    curr = queue.pop(0)
                    for w in adj[curr]:
                        if coloring[w] in (c, d) and w not in visited:
                            if w in neigh_d[1:]:
                                return False # Cycle detected
                            visited.add(w)
                            queue.append(w)
            return True
            
    def backtrack(idx, coloring):
        if idx == n:
            return True
        
        v = vertex_order[idx]
        
        # Try colors 0 to max_color-1
        # We can limit the range of colors to those used + 1
        max_used = max(coloring) + 1 if max(coloring) != -1 else 0
        # Actually, we should try up to the target K.
        # But we are iterating K in the outer loop.
        
        for c in range(0, max_used + 1):
            if is_valid_assignment(coloring, v, c):
                coloring[v] = c
                if backtrack(idx + 1, coloring):
                    return True
                coloring[v] = -1
        return False

    # Try increasing number of colors
    # We need to restrict the colors to 0..k-1
    # The above backtrack allows colors up to max_used.
    # We need to enforce that we only use k colors.
    
    def backtrack_k(idx, coloring, k):
        if idx == n:
            return True
        
        v = vertex_order[idx]
        
        for c in range(k):
            if is_valid_assignment(coloring, v, c):
                coloring[v] = c
                if backtrack_k(idx + 1, coloring, k):
                    return True
                coloring[v] = -1
        return False

    for k in range(2, n + 1):
        coloring = [-1] * n
        if backtrack_k(0, coloring, k):
            return [int(c) for c in coloring]
            
    return list(range(n))

# EVOLVE-BLOCK-END
