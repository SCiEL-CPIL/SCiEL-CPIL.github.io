"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each color pair, maintain a DSU to track connectivity
    # We'll use a dictionary: pair_dsu[(a,b)] = (parent_dict, rank_dict)
    
    def make_dsu(vertices):
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
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        return parent, rank, find, union
    
    # We'll track for each color pair the DSU state
    # Since we're doing backtracking, we need to be able to undo operations
    # This is complex, so let's use a simpler approach: for each assignment, 
    # just check if adding the new edges would create a cycle
    
    def would_create_cycle(coloring, v, c, adj):
        """Check if assigning color c to vertex v would create a bicolored cycle."""
        for u in adj[v]:
            if coloring[u] == -1:
                continue
            d = coloring[u]
            if d == c:
                continue
            # Check if u and v are already connected in the (c,d)-subgraph
            # via a path that doesn't use edge (u,v)
            # BFS from u, avoiding direct edge to v initially
            visited = set()
            stack = [u]
            visited.add(u)
            while stack:
                curr = stack.pop()
                if curr == v:
                    return True  # cycle would be created
                for w in adj[curr]:
                    if w == u and curr == v:
                        continue
                    if w == v and curr == u:
                        continue  # don't traverse the edge we're adding
                    if coloring[w] != -1 and coloring[w] in (c, d):
                        if w not in visited:
                            visited.add(w)
                            stack.append(w)
        return False
    
    def backtrack(coloring, idx, k):
        if idx == n:
            return True
        v = idx
        used_by_neighbors = set()
        for u in adj[v]:
            if coloring[u] != -1:
                used_by_neighbors.add(coloring[u])
        
        for c in range(k):
            if c in used_by_neighbors:
                continue
            if would_create_cycle(coloring, v, c, adj):
                continue
            coloring[v] = c
            if backtrack(coloring, idx + 1, k):
                return True
            coloring[v] = -1
        return False
    
    best_coloring = None
    for k in range(3, 8):
        coloring = [-1] * n
        if backtrack(coloring, 0, k):
            best_coloring = list(coloring)
            break
    
    if best_coloring is None:
        # Fallback greedy
        coloring = [0] * n
        for i in range(n):
            used = set()
            for u in adj[i]:
                if u < i:
                    used.add(coloring[u])
            for c in range(50):
                if c not in used:
                    coloring[i] = c
                    break
        best_coloring = list(coloring)
    
    return best_coloring

# EVOLVE-BLOCK-END
