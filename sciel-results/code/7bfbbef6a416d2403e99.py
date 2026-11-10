"""Rewritten program for bicolored-cycle-free coloring."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(coloring, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    # Group vertices by color
    color_to_vertices = defaultdict(list)
    for v in range(n):
        color_to_vertices[coloring[v]].append(v)
    
    colors = list(color_to_vertices.keys())
    
    # For each pair of colors, check if the bipartite subgraph is a forest
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            c1, c2 = colors[i], colors[j]
            verts1 = set(color_to_vertices[c1])
            verts2 = set(color_to_vertices[c2])
            verts = verts1 | verts2
            
            # Build subgraph on these vertices with edges between them
            # Use union-find to detect cycles
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False  # cycle detected
                parent[px] = py
                return True
            
            edge_count = 0
            for u, v in edges:
                if u in verts and v in verts:
                    edge_count += 1
                    if not union(u, v):
                        return False  # bicolored cycle found
            
            # A forest on |verts| vertices has at most |verts|-1 edges
            # But we already checked via union-find, so this is redundant
            # However, let's also verify: for a forest, edges <= vertices - 1
            # Actually union-find already catches cycles, so we're good
            pass
    
    return True

def _solve_with_k_colors(n, edges, k):
    """Try to find a valid coloring using exactly k colors (0..k-1)."""
    coloring = [-1] * n
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (all 4 here), but we can use a heuristic
    # Let's try vertex ordering by descending degree, then by index
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    # Backtracking
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        
        # Determine which colors are forbidden (neighbors' colors)
        forbidden = set()
        for u in adj[v]:
            if coloring[u] != -1:
                forbidden.add(coloring[u])
        
        for c in range(k):
            if c in forbidden:
                continue
            
            coloring[v] = c
            
            # Check if this creates a bicolored cycle
            # Only need to check pairs involving color c and any other color present
            # For efficiency, check incrementally
            valid = True
            other_colors = set()
            for u in adj[v]:
                if coloring[u] != -1 and coloring[u] != c:
                    other_colors.add(coloring[u])
            
            for oc in other_colors:
                # Check if adding v with color c creates a cycle in the (c, oc) subgraph
                # Find all vertices with color c or oc
                # Use union-find on the subgraph induced by colors c and oc
                parent = {}
                def find(x):
                    if x not in parent:
                        parent[x] = x
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    parent[px] = py
                    return True
                
                # Collect edges in the (c, oc) subgraph
                for u2, v2 in edges:
                    cu, cv = coloring[u2], coloring[v2]
                    if (cu == c and cv == oc) or (cu == oc and cv == c):
                        if not union(u2, v2):
                            valid = False
                            break
                    elif cu == c and cv == c:
                        # Same color, not in this bipartite subgraph
                        pass
                    elif cu == oc and cv == oc:
                        pass
                
                if not valid:
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s margin
    
    n = P['n']
    edges = P['edges']
    
    # Try increasing number of colors
    for k in range(2, n + 1):
        if time.time() > deadline:
            break
        result = _solve_with_k_colors(n, edges, k)
        if result is not None:
            # Verify
            if _check_bicolored_forest(result, n, edges):
                return list(result)
    
    # Fallback: use all different colors (always valid)
    return list(range(n))
