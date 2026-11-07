"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each pair of colors, maintain the edges between vertices of those colors
    # and use union-find to detect cycles incrementally
    
    def solve_with_k(k):
        """Try to find a valid coloring with at most k colors."""
        coloring = [None] * n
        
        # Union-Find for each color pair
        # We'll track for each (c1, c2) pair, the DSU of vertices colored c1 or c2
        # But that's expensive. Instead, we'll check incrementally.
        
        # Key insight: when we assign color c to vertex v, we only need to check
        # pairs (c, c') for all c' that have at least one neighbor of v with color c'
        
        def has_bicolored_cycle(v, c, coloring):
            """Check if adding vertex v with color c creates a bicolored cycle."""
            # For each other color c' present among neighbors
            for w in adj[v]:
                cw = coloring[w]
                if cw is None or cw == c:
                    continue
                # Check if adding v to the (c, cw) subgraph creates a cycle
                # The (c, cw) subgraph consists of all vertices colored c or cw
                # that are connected via edges within this set
                # A cycle is created if v has two neighbors in the same connected component
                # of the (c, cw) subgraph (excluding v)
                
                # Find connected components of the (c, cw) induced subgraph
                # (not including v)
                visited = set()
                comp_id = {}
                comp_counter = 0
                
                for u in range(n):
                    if u == v or coloring[u] is None:
                        continue
                    if coloring[u] not in (c, cw):
                        continue
                    if u in visited:
                        continue
                    # BFS
                    queue = [u]
                    visited.add(u)
                    comp_id[u] = comp_counter
                    while queue:
                        x = queue.pop(0)
                        for y in adj[x]:
                            if y == v:
                                continue
                            if coloring[y] is None:
                                continue
                            if coloring[y] not in (c, cw):
                                continue
                            if y not in visited:
                                visited.add(y)
                                comp_id[y] = comp_counter
                                queue.append(y)
                    comp_counter += 1
                
                # Now check if v has two neighbors in the same component
                seen_comps = set()
                for w in adj[v]:
                    if coloring[w] not in (c, cw):
                        continue
                    if w in comp_id:
                        if comp_id[w] in seen_comps:
                            return True
                        seen_comps.add(comp_id[w])
            
            return False
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = idx
            forbidden = set()
            for w in adj[v]:
                if coloring[w] is not None:
                    forbidden.add(coloring[w])
            
            for c in range(k):
                if c in forbidden:
                    continue
                coloring[v] = c
                if not has_bicolored_cycle(v, c, coloring):
                    if backtrack(idx + 1):
                        return True
                coloring[v] = None
            
            return False
        
        if backtrack(0):
            return coloring
        return None
    
    # Try increasing k
    for k in range(3, 15):
        result = solve_with_k(k)
        if result is not None:
            # Normalize colors to 0..k-1
            color_map = {}
            next_c = 0
            final = [0] * n
            for v in range(n):
                if result[v] not in color_map:
                    color_map[result[v]] = next_c
                    next_c += 1
                final[v] = color_map[result[v]]
            return final
    
    # Fallback
    return list(range(n))

# EVOLVE-BLOCK-END
