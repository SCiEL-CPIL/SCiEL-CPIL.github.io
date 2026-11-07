"""CSP-based coloring with incremental bicolored-forest detection."""
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start = time.time()
    deadline = start + 80.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Fallback: identity coloring
    best = list(range(n))
    best_cnt = n
    
    # Try k=3 first (best case), then k=4, k=5
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        # Map: position in order -> vertex
        pos_of = {v: i for i, v in enumerate(order)}
        
        # For each pair of colors, track DSU state incrementally
        # We'll use a backtracking search with constraint propagation
        
        coloring = [-1] * n
        
        # DSU for each color pair
        # We'll track: for each pair (c1,c2), a DSU over vertices
        # But that's expensive. Instead, we check bicolored cycles at the end
        # but prune early by checking if adding an edge creates a cycle in any pair.
        
        # Better approach: track for each color pair, the set of edges that connect
        # vertices of those two colors. A bicolored cycle exists iff any such set has a cycle.
        # We can maintain a DSU per color pair incrementally.
        
        # For k colors, there are k*(k-1)/2 pairs
        pairs = []
        pair_index = {}
        for i in range(k):
            for j in range(i+1, k):
                pair_index[(i,j)] = len(pairs)
                pairs.append((i,j))
        
        # DSU: parent array for each pair, but only over vertices that have colors in that pair
        # We'll use a simpler approach: for each pair, maintain a list of edges and check
        # at the end. But for pruning, we need incremental check.
        
        # Let's use a different strategy: assign colors in order, and for each assignment,
        # check if it creates a bicolored cycle with already-assigned neighbors.
        
        # For a vertex v being assigned color c:
        # For each neighbor u already assigned color d != c:
        #   The edge (u,v) is in the bicolored subgraph for pair (c,d).
        #   We need to check if this creates a cycle in that subgraph.
        
        # Maintain: for each color pair (c,d), a DSU over the vertices that have colors c or d
        # Actually, let's maintain for each pair a set of edges, and use a global check.
        
        # Simpler: use a recursive backtracking with a "last edge check" approach.
        # When we assign color c to vertex v, for each neighbor u with color d:
        #   Check if there's a path from u to v in the subgraph induced by colors {c,d}
        #   excluding the edge (u,v) itself. If yes, adding (u,v) creates a cycle.
        
        # This is O(n) per check, which is fine for 50 vertices.
        
        # Let's implement this:
        
        def has_path_in_pair(u, v, c1, c2, coloring, adj, ignore_edge=None):
            """Check if there's a path from u to v in the subgraph induced by colors c1,c2,
            ignoring edge (ignore_edge) if specified."""
            if u == v:
                return True
            visited = {u}
            stack = [u]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y in visited:
                        continue
                    if ignore_edge is not None and ((x == ignore_edge[0] and y == ignore_edge[1]) or (x == ignore_edge[1] and y == ignore_edge[0])):
                        continue
                    if coloring[y] in (c1, c2):
                        if y == v:
                            return True
                        visited.add(y)
                        stack.append(y)
            return False
        
        def backtrack(pos, coloring):
            if time.time() > deadline:
                return False
            if pos == len(order):
                return True
            
            v = order[pos]
            # Try each color
            for c in range(k):
                # Check proper coloring: no neighbor with same color
                valid = True
                for u in adj[v]:
                    if coloring[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check bicolored cycle: for each neighbor u with color d != c,
                # check if there's a path from u to v in subgraph {c,d} excluding edge (u,v)
                ok = True
                for u in adj[v]:
                    d = coloring[u]
                    if d == c or d == -1:
                        continue
                    # Check if adding edge (u,v) to the {c,d} subgraph creates a cycle
                    # This happens iff there's already a path from u to v in {c,d} subgraph
                    # excluding the edge (u,v)
                    if has_path_in_pair(u, v, c, d, coloring, adj, ignore_edge=(u, v)):
                        ok = False
                        break
                if not ok:
                    continue
                
                coloring[v] = c
                if backtrack(pos + 1, coloring):
                    return True
                coloring[v] = -1
            
            return False
        
        # Try multiple random orderings if the first fails
        import random
        rng = random.Random(42)
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            if attempt == 0:
                order = sorted(range(n), key=lambda v: -len(adj[v]))
            else:
                order = list(range(n))
                rng.shuffle(order)
            
            coloring = [-1] * n
            if backtrack(0, coloring):
                # Verify
                cnt = len(set(coloring))
                if cnt < best_cnt:
                    best = list(coloring)
                    best_cnt = cnt
                found = True
                break
            if time.time() > deadline:
                break
        
        if found:
            break
    
    return [int(c) for c in best]
