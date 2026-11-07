"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    deadline = time.time() + 55  # leave margin
    
    # Try increasing number of colors
    for k in range(3, 50):
        if time.time() > deadline - 5:
            break
        result = _solve_with_k_colors(n, adj, order, k, deadline)
        if result is not None:
            return result
    
    # Fallback: return a valid coloring with more colors
    result = _solve_with_k_colors(n, adj, order, 10, deadline)
    if result is not None:
        return result
    
    # Last resort: trivial coloring
    return list(range(n))


def _solve_with_k_colors(n, adj, order, k, deadline):
    """Try to color graph with exactly k colors (0..k-1) satisfying constraints."""
    color = [-1] * n  # -1 means unassigned
    
    # For each pair of colors (i, j), maintain union-find to detect cycles
    # in the subgraph induced by colors {i, j}
    
    # Union-find structures: uf[(i,j)] where i < j
    # Each union-find has parent array of size n
    
    uf_parents = {}  # (i,j) -> list of parents
    
    def get_uf(ci, cj):
        key = (min(ci, cj), max(ci, cj))
        if key not in uf_parents:
            uf_parents[key] = list(range(n))
        return uf_parents[key]
    
    def uf_find(parents, x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    
    def uf_union(parents, x, y):
        rx = uf_find(parents, x)
        ry = uf_find(parents, y)
        if rx == ry:
            return False  # cycle detected
        parents[rx] = ry
        return True
    
    # For efficiency, track for each color pair, which vertices are "active"
    
    # Backtracking with incremental union-find and undo
    # When assigning color c to vertex v:
    #   For each neighbor u of v with color cu:
    #     The edge (v, u) is in subgraph for pair {c, cu} (if c != cu)
    #     or in subgraphs for pairs {c, c'} for all c' if c == cu
    
    # We'll use a simpler approach: check validity by building subgraphs on the fly
    # For 50 vertices, this is fast enough
    
    def _is_forest_for_pair(c1, c2):
        """Check if subgraph induced by vertices with color c1 or c2 is a forest."""
        verts = set()
        for v in range(n):
            if color[v] == c1 or color[v] == c2:
                verts.add(v)
        
        if len(verts) <= 1:
            return True
        
        parent = {v: v for v in verts}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        for v in verts:
            for u in adj[v]:
                if u in verts and u > v:
                    if not union(v, u):
                        return False
        
        return True
    
    # Backtracking
    def backtrack(idx):
        if time.time() > deadline - 2:
            return None
        
        if idx == n:
            return list(color)
        
        v = order[idx]
        
        # Try colors 0 to k-1
        for c in range(k):
            # Check if color c is allowed for v (adjacent vertices must differ)
            ok = True
            for u in adj[v]:
                if color[u] == c:
                    ok = False
                    break
            
            if not ok:
                continue
            
            # Tentatively assign
            color[v] = c
            
            # Check bicolored forest constraints for all pairs involving c
            valid = True
            relevant_colors = set([c])
            for u in adj[v]:
                if color[u] != -1:
                    relevant_colors.add(color[u])
            
            for cu in relevant_colors:
                if cu == c:
                    continue
                if not _is_forest_for_pair(c, cu):
                    valid = False
                    break
            
            if not valid:
                color[v] = -1
                continue
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            color[v] = -1
        
        return None
    
    return backtrack(0)

# EVOLVE-BLOCK-END
