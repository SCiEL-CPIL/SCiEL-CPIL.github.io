"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles_full(coloring, adj):
    """Full check: verify no bicolored cycles using union-find per color pair."""
    n = P['n']
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair_edges[(cu, cv)].append((u, v))
        elif cv < cu:
            pair_edges[(cv, cu)].append((u, v))
    
    for (c1, c2), edges in pair_edges.items():
        parent = {}
        rank = {}
        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
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

def _backtrack_coloring(adj, num_colors, deadline, order=None):
    """Backtracking with incremental bicolored cycle checking."""
    n = P['n']
    coloring = [-1] * n
    
    if order is None:
        order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    def would_create_cycle(v, c, adj, coloring):
        neighbor_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1 and coloring[nb] != c:
                neighbor_colors.add(coloring[nb])
        
        for cn in neighbor_colors:
            target_neighbors = [nb for nb in adj[v] if coloring[nb] == cn]
            if not target_neighbors:
                continue
            
            visited = set([v])
            stack = [(v, -1)]
            found_cycle = False
            
            while stack and not found_cycle:
                cur, par = stack.pop()
                for nb in adj[cur]:
                    if nb == par:
                        continue
                    if coloring[nb] not in (c, cn):
                        continue
                    if nb == v:
                        found_cycle = True
                        break
                    if nb in visited:
                        if coloring[nb] == cn and nb in target_neighbors:
                            found_cycle = True
                            break
                        continue
                    visited.add(nb)
                    stack.append((nb, cur))
            
            if found_cycle:
                return False
        
        return True
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            
            coloring[v] = c
            if would_create_cycle(v, c, adj, coloring):
                result = dfs(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        
        return None
    
    return dfs(0)

def construct():
    deadline = time.time() + 85
    adj = _build_adj()
    
    for nc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, nc, deadline)
        if result is not None:
            if _check_bicolored_cycles_full(result, adj):
                return [int(x) for x in result]
    
    return [int(i) for i in range(P['n'])]
