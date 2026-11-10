"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 45.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list as tuples for faster lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # For each pair of colors, track which edges are "active" (both endpoints have those colors)
    # We need: for any color pair (c1,c2), the subgraph on vertices colored c1 or c2 is a forest
    
    def check_coloring(colors):
        """Check if a coloring is valid."""
        # Rule 1: adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Rule 2: for every pair of colors, induced subgraph is a forest
        color_to_verts = {}
        for i, c in enumerate(colors):
            if c not in color_to_verts:
                color_to_verts[c] = []
            color_to_verts[c].append(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_set = set(color_to_verts[c1]) | set(color_to_verts[c2])
                # Count edges and check forest property
                # A graph on m vertices is a forest iff it has at most m-1 edges AND is acyclic
                # Quick check: count edges first
                edge_count = 0
                for u, v in edges:
                    if u in verts_set and v in verts_set:
                        edge_count += 1
                m = len(verts_set)
                if edge_count >= m:  # If edges >= vertices, definitely has a cycle
                    return False
                # If edge_count < m, still need to check for cycles
                # Use union-find
                parent = {v: v for v in verts_set}
                rank = {v: 0 for v in verts_set}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                for u, v in edges:
                    if u in verts_set and v in verts_set:
                        if not union(u, v):
                            return False
        return True
    
    # Try increasing k from 2 upward
    # Use a smart backtracking with incremental forest checking
    
    def backtrack_with_forest_check(colors, k, vertex_order):
        """Backtracking that checks forest constraint incrementally."""
        n = len(colors)
        
        # For each color pair, maintain union-find state
        # This is complex; instead, do a full check at the end but prune early
        
        def is_valid_partial(colors, assigned_count):
            """Check if partial assignment can be extended to valid coloring."""
            # Check adjacency for assigned vertices
            for i in range(assigned_count):
                v = vertex_order[i]
                if colors[v] == -1:
                    continue
                for u in adj[v]:
                    if colors[u] != -1 and colors[u] == colors[v]:
                        return False
            
            # Check forest constraint for pairs of colors that are fully determined
            # This is expensive; do a lightweight check
            color_to_verts = {}
            for i in range(assigned_count):
                v = vertex_order[i]
                c = colors[v]
                if c == -1:
                    continue
                if c not in color_to_verts:
                    color_to_verts[c] = []
                color_to_verts[c].append(v)
            
            used_colors = sorted(color_to_verts.keys())
            for idx1 in range(len(used_colors)):
                for idx2 in range(idx1, len(used_colors)):
                    c1, c2 = used_colors[idx1], used_colors[idx2]
                    verts_set = set(color_to_verts[c1]) | set(color_to_verts[c2])
                    # Count edges among these vertices
                    edge_count = 0
                    for u, v in edges:
                        if u in verts_set and v in verts_set:
                            edge_count += 1
                    m = len(verts_set)
                    if edge_count >= m:
                        return False
            return True
        
        def solve(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = vertex_order[idx]
            for c in range(k):
                colors[v] = c
                if is_valid_partial(colors, idx + 1):
                    if solve(idx + 1):
                        return True
                colors[v] = -1
            return False
        
        return solve(0)
    
    # Order vertices: try to find a good ordering
    # Use degree-based ordering with tie-breaking by index
    adj_list = adj[:]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    # Remove duplicates
    for i in range(n):
        adj_list[i] = list(set(adj_list[i]))
    
    degree_order = sorted(range(n), key=lambda x: -len(adj_list[x]))
    
    best_result = None
    
    for k in range(2, 11):
        if time.time() > deadline:
            break
        
        colors = [-1] * n
        found = backtrack_with_forest_check(colors, k, degree_order)
        
        if found:
            # Verify the solution
            if check_coloring(colors):
                return [int(c) for c in colors]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
