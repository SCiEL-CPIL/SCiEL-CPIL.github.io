import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort vertices by degree descending for better pruning
    vertex_order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # For each pair of colors, track edges in that bicolored subgraph
    # We'll use union-find per color pair to detect cycles
    
    best_result = list(range(n))  # fallback: 50 colors
    
    def check_and_assign(coloring, v, c):
        """Try assigning color c to vertex v. Returns True if valid."""
        coloring[v] = c
        # Check all neighbors with same color -> not allowed
        for u in adj[v]:
            if coloring[u] == c:
                coloring[v] = -1
                return False
        # Check bicolored cycles: for each color d used by neighbors of v
        # the edges between v and neighbors of color d must not create a cycle
        # in the (c, d) bicolored subgraph
        for u in adj[v]:
            d = coloring[u]
            if d == c:
                continue
            # Adding edge (v, u) to the (c,d) bicolored subgraph
            # We need to check if this creates a cycle
            # The (c,d) subgraph has edges between vertices colored c and d
            # v is color c, u is color d
            # We need to check if v and u are already connected in the (c,d) forest
            # without using this edge
            # For efficiency, we track connectivity per color pair
            pass
        coloring[v] = -1
        return True
    
    # Better approach: use union-find per color pair
    # Since max colors will be small (maybe 4-6), we can afford this
    
    def solve_with_max_colors(max_c, coloring=None):
        if coloring is None:
            coloring = [-1] * n
        
        # Union-find structures for each color pair
        # parent[(c1, c2)] = list of n entries
        # We only track pairs where both colors are in use
        uf_parents = {}
        
        def find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(parent, a, b):
            ra, rb = find(parent, a), find(parent, b)
            if ra == rb:
                return False  # cycle detected
            parent[ra] = rb
            return True
        
        def get_uf(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in uf_parents:
                uf_parents[key] = list(range(n))
            return uf_parents[key]
        
        def remove_uf(c1, c2, u, v):
            """This is hard to do with union-find. Instead, we'll rebuild when needed."""
            pass
        
        # We'll use a different strategy: backtracking with full validation
        # at each step using a fresh union-find check
        
        def validate_partial(coloring, assigned):
            """Check if the partial coloring is valid for assigned vertices."""
            # Group vertices by color
            color_groups = defaultdict(list)
            for v in assigned:
                if coloring[v] >= 0:
                    color_groups[coloring[v]].append(v)
            
            # For each pair of colors, check the induced subgraph is a forest
            colors_used = list(color_groups.keys())
            for i in range(len(colors_used)):
                for j in range(i+1, len(colors_used)):
                    c1, c2 = colors_used[i], colors_used[j]
                    group = set(color_groups[c1]) | set(color_groups[c2])
                    # Build edges in this bicolored subgraph
                    uf = list(range(n))
                    def find2(x):
                        while uf[x] != x:
                            uf[x] = uf[uf[x]]
                            x = uf[x]
                        return x
                    def union2(a, b):
                        ra, rb = find2(a), find2(b)
                        if ra == rb:
                            return False
                        uf[ra] = rb
                        return True
                    for v in color_groups[c1]:
                        for u in adj[v]:
                            if u in color_groups[c2] and u > v:  # avoid double counting
                                if not union2(v, u):
                                    return False
            return True
        
        # This is too slow. Let me use incremental union-find.
        # Strategy: maintain union-find per color pair incrementally.
        # When we assign a color to a vertex, we add edges to the relevant color pairs.
        # When we unassign, we need to remove - which is hard.
        # Alternative: use a counter-based approach.
        
        # Let me try a cleaner backtracking with efficient cycle checking.
        
        def backtrack(idx, coloring, uf_states):
            """
            idx: index in vertex_order
            coloring: current assignment
            uf_states: dict mapping (c1,c2) -> (parent_list, edge_count)
                       We'll use a simpler approach: just revalidate periodically
            """
            if time.time() > deadline:
                return None
            
            if idx == n:
                # All assigned, return coloring
                return list(coloring)
            
            v = vertex_order[idx]
            
            # Try each color
            for c in range(max_c):
                # Check neighbor constraint
                valid = True
                for u in adj[v]:
                    if coloring[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[v] = c
                
                # Check bicolored cycles involving v
                # For each neighbor u with color d != c, adding edge (v,u) 
                # to the (c,d) subgraph must not create a cycle
                cycle_found = False
                for u in adj[v]:
                    d = coloring[u]
                    if d == c or d < 0:
                        continue
                    # Check if v and u are connected in (c,d) subgraph without this edge
                    # We need to do a BFS/DFS in the (c,d) induced subgraph
                    # excluding the edge (v,u)
                    # This is O(E) per check, but n=50 so it's fine
                    # Actually, let's use a simpler check:
                    # In the (c,d) subgraph, v is color c, u is color d
                    # They are connected if there's a path from v to u using only
                    # vertices of color c or d, not using the direct edge (v,u)
                    # Since v and u are adjacent, we check if there's another path.
                    
                    # BFS from v in (c,d) subgraph, not using edge (v,u)
                    visited = {v}
                    queue = [v]
                    while queue:
                        cur = queue.pop()
                        for w in adj[cur]:
                            if w == u and cur == v:
                                continue  # skip the direct edge we're adding
                            if w == v and cur == u:
                                continue
                            if w in visited:
                                continue
                            cw = coloring[w]
                            if cw == c or cw == d:
                                visited.add(w)
                                queue.append(w)
                    if u in visited:
                        cycle_found = True
                        break
                
                if cycle_found:
                    coloring[v] = -1
                    continue
                
                # Recurse
                result = backtrack(idx + 1, coloring, uf_states)
                if result is not None:
                    return result
                
                coloring[v] = -1
            
            return None
        
        coloring = [-1] * n
        result = backtrack(0, coloring, {})
        return result
    
    # Try with increasing number of colors
    for max_c in range(3, 10):
        if time.time() > deadline:
            break
        result = solve_with_max_colors(max_c)
        if result is not None:
            # Validate the result
            # Check proper coloring
            valid = True
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
            if valid:
                # Check bicolored forest constraint
                from collections import defaultdict as dd
                color_groups = dd(list)
                for i, c in enumerate(result):
                    color_groups[c].append(i)
                
                colors_used = sorted(color_groups.keys())
                forest_ok = True
                for i in range(len(colors_used)):
                    for j in range(i+1, len(colors_used)):
                        c1, c2 = colors_used[i], colors_used[j]
                        group = set(color_groups[c1]) | set(color_groups[c2])
                        uf = list(range(n))
                        def find(x):
                            while uf[x] != x:
                                uf[x] = uf[uf[x]]
                                x = uf[x]
                            return x
                        def union(a, b):
                            ra, rb = find(a), find(b)
                            if ra == rb:
                                return False
                            uf[ra] = rb
                            return True
                        for v in color_groups[c1]:
                            for u in adj[v]:
                                if u in color_groups[c2] and u > v:
                                    if not union(v, u):
                                        forest_ok = False
                                        break
                            if not forest_ok:
                                break
                        if not forest_ok:
                            break
                    if not forest_ok:
                        break
                
                if forest_ok:
                    return [int(x) for x in result]
        
        best_result = result if result is not None else best_result
    
    # Fallback: return the best we have or a valid 50-color solution
    return [int(x) for x in best_result]
