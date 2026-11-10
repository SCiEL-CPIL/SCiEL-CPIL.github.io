"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    es = set()
    for e in edges:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_max_clique():
        """Branch and bound with greedy coloring upper bound."""
        best_clique = []
        best_size = 0
        
        def greedy_color(order):
            """Color vertices in given order, return color class sizes."""
            color_classes = []
            for v in order:
                placed = False
                for ci, cc in enumerate(color_classes):
                    if not any((v, u) in es or (u, v) in es for u in cc):
                        cc.append(v)
                        placed = True
                        break
                if not placed:
                    color_classes.append([v])
            return len(color_classes)
        
        def expand(R, C):
            nonlocal best_clique, best_size
            if len(R) + len(C) <= best_size:
                return
            if not C:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return
            
            # Greedy coloring of C to get upper bound
            # Sort C by degree in C for better coloring
            C_sorted = sorted(C, key=lambda v: len(adj[v] & set(C)))
            colors = [0] * len(C_sorted)
            color_class_sizes = []
            for i, v in enumerate(C_sorted):
                used_colors = set()
                for j in range(i):
                    if (v, C_sorted[j]) in es or (C_sorted[j], v) in es:
                        used_colors.add(colors[j])
                c = 0
                while c in used_colors:
                    c += 1
                colors[i] = c
                if c == len(color_class_sizes):
                    color_class_sizes.append(1)
                else:
                    color_class_sizes[c] += 1
            
            # Upper bound: len(R) + max color
            ub = len(R) + len(color_class_sizes)
            if ub <= best_size:
                return
            
            # Try vertices in reverse color order (higher colors first)
            # Process from last color class to first
            for ci in range(len(color_class_sizes) - 1, -1, -1):
                if len(R) + ci + 1 <= best_size:
                    break
                # Find vertices with this color
                verts = [C_sorted[i] for i in range(len(C_sorted)) if colors[i] == ci]
                for v in reversed(verts):
                    if len(R) + 1 <= best_size:
                        break
                    new_R = R + [v]
                    new_C = [u for u in C if u != v and ((u, v) in es or (v, u) in es)]
                    expand(new_R, new_C)
        
        all_vertices = list(range(n))
        expand([], all_vertices)
        return best_clique
    
    # Find maximum clique
    clique = find_max_clique()
    
    if len(clique) < 2:
        # Fallback: find any edge
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    # For each pair of branch vertices, find a simple path between them
    # that avoids all other branch vertices in its interior
    branches = sorted(clique)
    branch_set = set(branches)
    paths = {}
    
    for pair in combinations(branches, 2):
        u, v = pair
        # BFS for shortest path from u to v avoiding other branches in interior
        # Interior vertices must not be in branch_set
        from collections import deque
        parent = {}
        parent[u] = None
        queue = deque([u])
        found = False
        while queue and not found:
            if time.time() > deadline:
                break
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == v:
                    # Found path
                    path = [v]
                    p = cur
                    while p is not None:
                        path.append(p)
                        p = parent[p]
                    path.reverse()
                    paths[pair] = path
                    found = True
                    break
                if nb not in parent:
                    # Check if nb is a branch vertex (not u or v)
                    if nb in branch_set and nb != u and nb != v:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        if not found:
            # Fallback: direct edge if exists
            if (u, v) in es or (v, u) in es:
                paths[pair] = [u, v]
            else:
                # Should not happen in a clique, but handle gracefully
                paths[pair] = [u, v]
    
    # Build final paths dict
    result_paths = []
    for pair in combinations(branches, 2):
        if pair in paths:
            result_paths.append([int(x) for x in paths[pair]])
        else:
            result_paths.append([int(pair[0]), int(pair[1])])
    
    return {'branches': [int(b) for b in branches], 'paths': result_paths}
# EVOLVE-BLOCK-END
