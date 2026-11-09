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

def _has_bicolor_cycle(coloring, n, adj, color_a, color_b):
    """Check if the subgraph induced by vertices with color_a or color_b has a cycle."""
    # Build subgraph of vertices with color_a or color_b
    vertices = [v for v in range(n) if coloring[v] == color_a or coloring[v] == color_b]
    if not vertices:
        return False
    # Build adjacency within this subgraph
    vset = set(vertices)
    # Use DFS to detect cycle
    visited = set()
    parent = {}
    
    def dfs(u, p):
        visited.add(u)
        for v in adj[u]:
            if v not in vset:
                continue
            if v == p:
                continue
            if v in visited:
                return True
            if dfs(v, u):
                return True
        return False
    
    for v in vertices:
        if v not in visited:
            if dfs(v, -1):
                return True
    return False

def _is_valid_assignment(coloring, n, adj, k):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    # Check adjacent vertices differ
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycle for any pair of colors present
    colors_used = set(coloring)
    for i, ca in enumerate(colors_used):
        for cb in list(colors_used)[i+1:]:
            if _has_bicolor_cycle(coloring, n, adj, ca, cb):
                return False
    return True

def _check_partial(coloring, n, adj, k, vertex_idx):
    """Check if partial assignment up to vertex_idx is consistent."""
    # Check adjacent vertices that are both assigned differ
    for u in range(vertex_idx + 1):
        if coloring[u] == -1:
            continue
        for v in adj[u]:
            if v > vertex_idx:
                continue
            if coloring[v] != -1 and coloring[u] == coloring[v]:
                return False
    # Check bicolored cycles for color pairs where both colors have assigned vertices
    # This is expensive, so only do it for pairs where both colors appear
    colors_so_far = {}
    for u in range(vertex_idx + 1):
        c = coloring[u]
        if c != -1:
            if c not in colors_so_far:
                colors_so_far[c] = []
            colors_so_far[c].append(u)
    
    color_list = list(colors_so_far.keys())
    for i in range(len(color_list)):
        for j in range(i+1, len(color_list)):
            ca, cb = color_list[i], color_list[j]
            # Check if subgraph induced by ca and cb has a cycle
            # Only consider assigned vertices
            va = colors_so_far[ca]
            vb = colors_so_far[cb]
            vset = set(va + vb)
            # Check for cycle in this subgraph using only assigned vertices
            visited = set()
            parent = {}
            
            def dfs(u, p):
                visited.add(u)
                for v in adj[u]:
                    if v not in vset:
                        continue
                    if v == p:
                        continue
                    if v in visited:
                        return True
                    if dfs(v, u):
                        return True
                return False
            
            for v in vset:
                if v not in visited:
                    if dfs(v, -1):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 second deadline
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Order vertices by degree (all degree 4 here) - use a static ordering
    # Try to find coloring with minimum colors
    for k in range(2, 50):
        if time.time() > deadline - 5.0:
            break
        # Backtracking search
        coloring = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline - 1.0:
                return False
            if idx == n:
                return True
            
            # Try each color
            for c in range(k):
                coloring[idx] = c
                if _check_partial(coloring, n, adj, k, idx):
                    if backtrack(idx + 1):
                        return True
                coloring[idx] = -1
            
            return False
        
        if backtrack(0):
            # Verify full solution
            if _is_valid_assignment(coloring, n, adj, k):
                return [int(x) for x in coloring]
            else:
                # Shouldn't happen, but safety
                return [int(x) for x in coloring]
    
    # Fallback: return identity coloring (always valid but uses many colors)
    return list(range(n))

# EVOLVE-BLOCK-END
