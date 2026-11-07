"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolored_cycle(n, colors, k):
    """Check if any pair of colors induces a cycle using union-find."""
    # For each pair of colors, build the subgraph and check for cycles
    color_sets = defaultdict(list)
    for i in range(n):
        color_sets[colors[i]].append(i)
    
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Get all vertices with color c1 or c2
            verts = set(color_sets[c1] + color_sets[c2])
            if len(verts) < 3:
                continue
            # Build subgraph and check for cycles using union-find
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return False
            
            for u in verts:
                for v in P['edges']:
                    if v[0] == u and v[1] in verts:
                        if union(u, v[1]):
                            return True
    return False

def _is_valid_coloring(n, colors, k):
    """Check proper coloring and bicolored forest constraint."""
    # Check proper coloring
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forest constraint
    if _has_bicolored_cycle(n, colors, k):
        return False
    return True

def _backtrack_search(n, edges, k, deadline):
    """Backtracking search for a valid k-coloring."""
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Precompute for each vertex, which colors are forbidden by neighbors
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return colors[:]
        v = order[idx]
        # Determine forbidden colors from already-colored neighbors
        forbidden = set()
        for u in adj[v]:
            if colors[u] != -1:
                forbidden.add(colors[u])
        
        for c in range(k):
            if c in forbidden:
                continue
            colors[v] = c
            # Check if this creates a bicolored cycle
            # Only need to check pairs involving color c
            cycle_found = False
            for c2 in range(k):
                if c2 == c:
                    continue
                # Check subgraph induced by colors c and c2
                verts = set()
                for i in range(n):
                    if colors[i] == c or colors[i] == c2:
                        verts.add(i)
                if len(verts) < 3:
                    continue
                parent = {vv: vv for vv in verts}
                rank = {vv: 0 for vv in verts}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return True
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return False
                
                for u in verts:
                    for w in adj[u]:
                        if w in verts:
                            if union(u, w):
                                cycle_found = True
                                break
                    if cycle_found:
                        break
                if cycle_found:
                    break
            
            if not cycle_found:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            colors[v] = -1
        return None
    
    return backtrack(0)

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 75  # Leave 15s margin
    
    # Try increasing numbers of colors
    for k in range(2, 50):
        result = _backtrack_search(n, edges, k, deadline)
        if result is not None:
            # Verify
            if _is_valid_coloring(n, result, k):
                return [int(c) for c in result]
    
    # Fallback: return a valid but potentially non-minimal coloring
    # Use a greedy approach with many colors
    colors = [0] * n
    adj = _build_adjacency(n, edges)
    for v in range(n):
        forbidden = set(colors[u] for u in adj[v])
        c = 0
        while c in forbidden:
            c += 1
        colors[v] = c
    
    # This may not satisfy bicolored constraint, but it's a fallback
    # Actually, let's just return unique colors to be safe
    return list(range(n))
# EVOLVE-BLOCK-END

