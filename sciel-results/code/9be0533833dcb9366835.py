"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolor_cycle(coloring, n, adj, color_a, color_b):
    """Check if the subgraph induced by vertices with color_a or color_b has a cycle using union-find."""
    vertices = [v for v in range(n) if coloring[v] == color_a or coloring[v] == color_b]
    if not vertices:
        return False
    vset = set(vertices)
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}
    
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
    
    for u in vertices:
        for v in adj[u]:
            if v in vset and u < v:
                if union(u, v):
                    return True
    return False

def _is_valid_assignment(coloring, n, adj):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = list(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            if _has_bicolor_cycle(coloring, n, adj, colors_used[i], colors_used[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Precompute vertex degrees for ordering
    degrees = [len(adj[v]) for v in range(n)]
    
    # Try increasing number of colors
    for k in range(2, 8):
        if time.time() > deadline - 3.0:
            break
        
        coloring = [-1] * n
        unassigned = set(range(n))
        assigned_count = 0
        
        # For MRV: track which colors are forbidden for each unassigned vertex
        # forbidden[v] = set of colors that cannot be used for v
        forbidden = [set() for _ in range(n)]
        
        # Precompute adjacency lists as sorted lists for speed
        adj_list = [sorted(adj[v]) for v in range(n)]
        
        def backtrack():
            nonlocal assigned_count
            if time.time() > deadline - 1.0:
                return False
            
            if assigned_count == n:
                return True
            
            # MRV: pick unassigned vertex with fewest available colors
            best_v = -1
            best_count = k + 1
            for v in unassigned:
                avail = k - len(forbidden[v])
                if avail < best_count:
                    best_count = avail
                    best_v = v
                    if avail <= 1:
                        break
            
            if best_v == -1 or best_count <= 0:
                return False
            
            v = best_v
            unassigned.remove(v)
            
            # Try each available color
            for c in range(k):
                if c in forbidden[v]:
                    continue
                
                coloring[v] = c
                assigned_count += 1
                
                # Check bicolored cycles involving color c
                # Only need to check pairs (c, other) where other is a color used by assigned vertices
                valid = True
                # Find all colors used by assigned vertices
                used_colors = set()
                for u in range(n):
                    if coloring[u] != -1:
                        used_colors.add(coloring[u])
                
                # Check all pairs involving c
                for other in used_colors:
                    if other == c:
                        continue
                    if _has_bicolor_cycle(coloring, n, adj, c, other):
                        valid = False
                        break
                
                # Also check pairs not involving c that might now be affected
                # Actually, assigning a new vertex with color c only creates new edges
                # in the subgraph for pairs involving c. Pairs not involving c are unchanged.
                # So we only need to check pairs involving c.
                
                if valid:
                    # Update forbidden sets for neighbors
                    changed = []
                    for nb in adj_list[v]:
                        if nb in unassigned:
                            forbidden[nb].add(c)
                            changed.append(nb)
                    
                    if backtrack():
                        return True
                    
                    # Undo
                    for nb in changed:
                        forbidden[nb].discard(c)
                
                coloring[v] = -1
                assigned_count -= 1
            
            unassigned.add(v)
            return False
        
        if backtrack():
            # Verify
            if _is_valid_assignment(coloring, n, adj):
                return [int(x) for x in coloring]
    
    # Fallback: greedy coloring with many colors
    coloring = [0] * n
    for v in range(n):
        used = set()
        for nb in adj_list[v]:
            if nb < v:
                used.add(coloring[nb])
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    
    # Verify fallback
    if _is_valid_assignment(coloring, n, adj):
        return [int(x) for x in coloring]
    
    # Ultimate fallback: identity
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
