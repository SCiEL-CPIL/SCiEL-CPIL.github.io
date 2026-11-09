"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected_in_induced(u, v, c1, c2, coloring, adj):
    """
    Check if u and v are connected in the subgraph induced by colors c1 and c2,
    excluding the direct edge (u, v).
    """
    if u == v:
        return False
    
    visited = {u}
    queue = [u]
    
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if neighbor == v and curr == u:
                continue
            if neighbor == u and curr == v:
                continue
            
            # Check if neighbor is in the induced subgraph
            if coloring[neighbor] == c1 or coloring[neighbor] == c2:
                if neighbor not in visited:
                    if neighbor == v:
                        return True
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return False

def _check_assignment_valid(v, c, coloring, adj, k):
    """
    Check if assigning color c to vertex v is valid given current coloring.
    1. No neighbor has color c.
    2. For each neighbor u with color d != c, adding edge (u, v) to G[{c, d}] must not create a cycle.
       This is equivalent to u and v NOT being connected in G[{c, d}] \ {(u, v)}.
    """
    for u in adj[v]:
        if coloring[u] == c:
            return False
        
        d = coloring[u]
        if d == -1:
            continue
        
        # Check cycle in G[{c, d}]
        # If u and v are already connected via other paths in G[{c, d}], adding (u, v) creates a cycle.
        if _is_connected_in_induced(u, v, c, d, coloring, adj):
            return False
            
    return True

def _get_mrv_vertex(coloring, adj, k):
    """
    Find the uncolored vertex with the fewest valid colors.
    Returns (vertex_index, valid_colors_list)
    """
    best_v = -1
    best_count = float('inf')
    best_colors = []
    
    for v in range(len(coloring)):
        if coloring[v] != -1:
            continue
            
        valid_colors = []
        for c in range(k):
            if _check_assignment_valid(v, c, coloring, adj, k):
                valid_colors.append(c)
                
        count = len(valid_colors)
        if count < best_count:
            best_count = count
            best_v = v
            best_colors = valid_colors
            
            if count == 0:
                return v, [] # Dead end
                
    return best_v, best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Try k from 3 to 5
    for k in range(3, 6):
        if time.time() > deadline - 5.0:
            break
            
        coloring = [-1] * n
        
        def backtrack():
            if time.time() > deadline - 1.0:
                return False
            
            # Check if all colored
            if all(c != -1 for c in coloring):
                return True
                
            v, valid_colors = _get_mrv_vertex(coloring, adj, k)
            
            if v == -1:
                # Should not happen if all are colored, but safety
                return True
                
            if not valid_colors:
                return False
                
            for c in valid_colors:
                coloring[v] = c
                if backtrack():
                    return True
                coloring[v] = -1
                
            return False

        if backtrack():
            # Verify solution
            # (The backtracking ensures validity, but a final check is good practice)
            is_valid = True
            for u, v in P['edges']:
                if coloring[u] == coloring[v]:
                    is_valid = False
                    break
            if is_valid:
                # Check bicolored cycles for all pairs
                colors_used = set(coloring)
                for i, ca in enumerate(colors_used):
                    for cb in list(colors_used)[i+1:]:
                        # Check if G[{ca, cb}] has a cycle
                        # We can reuse a simple DFS cycle check for the final verification
                        # Or trust the incremental check. Let's do a quick check.
                        subgraph_vertices = [v for v in range(n) if coloring[v] in (ca, cb)]
                        subgraph_adj = {v: [] for v in subgraph_vertices}
                        for u in subgraph_vertices:
                            for w in adj[u]:
                                if w in subgraph_vertices:
                                    subgraph_adj[u].append(w)
                        
                        visited = set()
                        def dfs_cycle(u, parent):
                            visited.add(u)
                            for w in subgraph_adj[u]:
                                if w == parent:
                                    continue
                                if w in visited:
                                    return True
                                if dfs_cycle(w, u):
                                    return True
                            return False
                            
                        for v in subgraph_vertices:
                            if v not in visited:
                                if dfs_cycle(v, -1):
                                    is_valid = False
                                    break
                        if not is_valid:
                            break
                            
            if is_valid:
                return [int(x) for x in coloring]
                
    # Fallback
    return list(range(n))

# EVOLVE-BLOCK-END
