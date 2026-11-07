import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edge_list = P['edges']

adj = defaultdict(list)
for u, v in edge_list:
    adj[u].append(v)
    adj[v].append(u)

# Vertex ordering by degree (descending) for backtracking
vertices_by_deg = sorted(range(N), key=lambda x: -len(adj[x]))
# Map from vertex to its position in the ordering
pos_of = {v: i for i, v in enumerate(vertices_by_deg)}

# For each vertex, precompute neighbors that come AFTER it in the ordering
# (these are the "forward" neighbors we need to check for same-color conflicts)
forward_neighbors = {}
for v in range(N):
    forward_neighbors[v] = [u for u in adj[v] if pos_of[u] > pos_of[v]]

def construct():
    deadline = time.time() + 85.0
    
    for K in range(3, 8):
        # Backtracking search for K-coloring with bicolored-forest constraint
        colors = [-1] * N
        # DSU for each color pair
        # We'll use a dict: (min_color, max_color) -> (parent_list, rank_list)
        dsu_parents = {}
        dsu_ranks = {}
        # Track which edges are "active" in each pair (for undo)
        # Actually, we'll just snapshot the DSU state before each assignment
        
        def find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(parent, rank, x, y):
            rx = find(parent, x)
            ry = find(parent, y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        # Snapshot/restore DSU state
        def dsu_snapshot():
            snap = {}
            for key in dsu_parents:
                snap[key] = (list(dsu_parents[key]), list(dsu_ranks[key]))
            return snap
        
        def dsu_restore(snap):
            # Remove keys not in snap
            for key in list(dsu_parents.keys()):
                if key not in snap:
                    del dsu_parents[key]
                    del dsu_ranks[key]
            for key, (p, r) in snap.items():
                dsu_parents[key] = p
                dsu_ranks[key] = r
        
        found = [False]
        
        def backtrack(idx):
            if found[0]:
                return True
            if time.time() > deadline:
                return False
            if idx == N:
                found[0] = True
                return True
            
            v = vertices_by_deg[idx]
            snap = dsu_snapshot()
            
            for c in range(K):
                # Check same-color constraint with forward neighbors
                ok = True
                for u in forward_neighbors[v]:
                    if colors[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Also check same-color with backward neighbors that are already colored
                # Actually we need to check ALL neighbors, not just forward
                # But backward neighbors are already colored, so check them too
                for u in adj[v]:
                    if colors[u] == c and pos_of[u] < pos_of[v]:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Assign color c to v
                colors[v] = c
                
                # Check bicolored cycle constraint with all colored neighbors
                valid = True
                # For each neighbor u that is already colored (pos_of[u] < pos_of[v])
                # and has a different color, check if adding edge (v,u) creates a cycle
                for u in adj[v]:
                    cu = colors[u]
                    if cu == -1 or cu == c:
                        continue
                    # Edge between v (color c) and u (color cu)
                    # This edge is in pair (min(c,cu), max(c,cu))
                    a, b = (c, cu) if c < cu else (cu, c)
                    key = (a, b)
                    if key not in dsu_parents:
                        dsu_parents[key] = list(range(N))
                        dsu_ranks[key] = [0] * N
                    if not union(dsu_parents[key], dsu_ranks[key], v, u):
                        valid = False
                        break
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                colors[v] = -1
                dsu_restore(snap)
            
            return False
        
        if backtrack(0):
            result = [int(c) for c in colors]
            # Validate
            for u, v in edge_list:
                if result[u] == result[v]:
                    break
            else:
                # Check bicolored cycles
                valid = True
                used = set(result)
                for i in range(len(used)):
                    for j in range(i + 1, len(used)):
                        a, b = list(used)[i], list(used)[j]
                        parent = list(range(N))
                        rank = [0] * N
                        for u, v in edge_list:
                            cu, cv = result[u], result[v]
                            if (cu == a and cv == b) or (cu == b and cv == a):
                                # Union
                                def f(x):
                                    while parent[x] != x:
                                        parent[x] = parent[parent[x]]
                                        x = parent[x]
                                    return x
                                rx, ry = f(u), f(v)
                                if rx == ry:
                                    valid = False
                                    break
                                if rank[rx] < rank[ry]:
                                    rx, ry = ry, rx
                                parent[ry] = rx
                                if rank[rx] == rank[ry]:
                                    rank[rx] += 1
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return result
            # If validation failed, continue to next K
        # If backtracking didn't find a solution or timed out, try next K
    
    # Ultimate fallback: proper coloring with N colors
    return list(range(N))

# EVOLVE-BLOCK-END
