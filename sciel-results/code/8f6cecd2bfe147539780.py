"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles_full(coloring, adj):
    """Full check using union-find per color pair. Returns True if valid."""
    n = len(coloring)
    # Check adjacent same color
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    
    # Group edges by color pair
    color_pairs = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                cu, cv = coloring[u], coloring[v]
                key = (min(cu, cv), max(cu, cv))
                if key not in color_pairs:
                    color_pairs[key] = []
                color_pairs[key].append((u, v))
    
    for (c1, c2), edges in color_pairs.items():
        # Union-find for vertices involved in this color pair
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        
        for u, v in edges:
            if not union(u, v):
                return False
    return True

def _creates_bicolored_cycle(coloring, adj, v, c, nb, cn):
    """Check if adding vertex v with color c creates a bicolored cycle 
    in the subgraph induced by colors {c, cn} via edge (v, nb).
    Returns True if a cycle is created."""
    # BFS from v in the {c, cn} subgraph, trying to reach nb
    # Skip the direct edge (v, nb)
    visited = set()
    stack = [v]
    visited.add(v)
    
    while stack:
        cur = stack.pop()
        for nb2 in adj[cur]:
            if nb2 == nb and cur == v:
                continue  # skip the edge we're adding
            if coloring[nb2] in (c, cn) and nb2 not in visited:
                if nb2 == nb:
                    return True  # found path back to nb, cycle created
                visited.add(nb2)
                stack.append(nb2)
    return False

def _backtrack_coloring(adj, num_colors, deadline):
    """Backtracking with targeted bicolored-cycle checking."""
    n = P['n']
    coloring = [-1] * n
    
    # Use a snake-like ordering for this structured graph
    # The graph appears to be a 5x10 or similar grid with wraparound
    # Let's use natural order first, then try alternatives
    order = list(range(n))
    
    nodes_explored = [0]
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        nodes_explored[0] += 1
        
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        
        # Get colors used by already-colored neighbors
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        
        # Try each color not used by neighbors
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            
            coloring[v] = c
            
            # Targeted check: for each colored neighbor, verify no bicolored cycle
            valid = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1:
                    continue
                # Check if adding edge (v, nb) creates a cycle in {c, cn} subgraph
                if _creates_bicolored_cycle(coloring, adj, v, c, nb, cn):
                    valid = False
                    break
            
            if valid:
                result = dfs(idx + 1)
                if result is not None:
                    return result
            
            coloring[v] = -1
        
        return None
    
    return dfs(0)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    adj = _build_adj()
    
    # Try with increasing number of colors
    # 3 colors is the theoretical minimum for this graph
    for nc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, nc, deadline)
        if result is not None:
            # Final verification
            if _check_bicolored_cycles_full(result, adj):
                return [int(x) for x in result]
    
    # Fallback: use all distinct colors (always valid)
    return [int(i) for i in range(P['n'])]
