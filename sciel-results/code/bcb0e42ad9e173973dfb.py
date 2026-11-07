import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class ReversibleUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.undo_stack = []
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.undo_stack.append((rx, ry, self.parent[ry], self.rank[rx], self.rank[ry]))
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
    
    def undo(self):
        if not self.undo_stack:
            return
        rx, ry, old_parent_ry, old_rank_rx, old_rank_ry = self.undo_stack.pop()
        self.parent[ry] = old_parent_ry
        self.rank[rx] = old_rank_rx
        self.rank[ry] = old_rank_ry
    
    def snapshot(self):
        return len(self.undo_stack)
    
    def rollback(self, snap):
        while len(self.undo_stack) > snap:
            self.undo()

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def is_valid_coloring(col):
        """Check if a complete coloring is valid."""
        for u, v in edges:
            if col[u] == col[v]:
                return False
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = col[u], col[v]
            if cu != cv:
                pair = (min(cu, cv), max(cu, cv))
                pair_edges[pair].append((u, v))
        for pair, elist in pair_edges.items():
            uf = ReversibleUnionFind(n)
            for u, v in elist:
                if not uf.union(u, v):
                    return False
        return True
    
    def solve_with_k(k, time_limit):
        """Solve using MRV + forward checking with union-find for bicolored cycles."""
        coloring = [-1] * n
        # For each color pair, track which vertices are connected
        pair_ufs = {}
        
        def get_uf(c1, c2):
            pair = (min(c1, c2), max(c1, c2))
            if pair not in pair_ufs:
                pair_ufs[pair] = ReversibleUnionFind(n)
            return pair_ufs[pair]
        
        def assign(v, c):
            """Assign color c to vertex v. Returns list of (uf, snapshot) or None if conflict."""
            # Check no neighbor has same color
            for nb in adj[v]:
                if coloring[nb] == c:
                    return None
            
            coloring[v] = c
            snapshots = []
            
            for nb in adj[v]:
                d = coloring[nb]
                if d == -1:
                    continue
                uf = get_uf(c, d)
                snap = uf.snapshot()
                if not uf.union(v, nb):
                    coloring[v] = -1
                    for ufk, sn in snapshots:
                        ufk.rollback(sn)
                    return None
                snapshots.append((uf, snap))
            return snapshots
        
        def unassign(v, snapshots):
            """Undo assignment of vertex v."""
            coloring[v] = -1
            for uf, snap in snapshots:
                uf.rollback(snap)
        
        def get_available_colors(v):
            """Get colors that don't conflict with neighbors."""
            used = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            return [c for c in range(k) if c not in used]
        
        def mrv_select():
            """Select uncolored vertex with minimum remaining values (available colors)."""
            best_v = -1
            best_count = k + 1
            for v in range(n):
                if coloring[v] == -1:
                    avail = get_available_colors(v)
                    if len(avail) < best_count:
                        best_count = len(avail)
                        best_v = v
                        if best_count == 0:
                            return v, avail
            return best_v, get_available_colors(best_v) if best_v != -1 else []
        
        # Symmetry breaking: fix vertex 0 to color 0
        coloring[0] = 0
        
        def backtrack():
            if time.time() > time_limit:
                return False
            
            v, avail = mrv_select()
            if v == -1:
                return True  # All colored
            
            if not avail:
                return False
            
            # Order colors by least constraining value heuristic
            # Try colors that appear least among neighbors' neighbors
            def color_score(c):
                cnt = 0
                for nb in adj[v]:
                    if coloring[nb] == -1:
                        # Count how many of nb's other neighbors use color c
                        for nb2 in adj[nb]:
                            if nb2 != v and coloring[nb2] == c:
                                cnt += 1
                return cnt
            
            avail.sort(key=color_score)
            
            for c in avail:
                snapshots = assign(v, c)
                if snapshots is not None:
                    if backtrack():
                        return True
                    unassign(v, snapshots)
            return False
        
        if backtrack():
            return coloring
        return None
    
    # Try increasing k
    for k in range(3, 15):
        if time.time() > deadline - 5:
            break
        result = solve_with_k(k, deadline - 5)
        if result is not None:
            if is_valid_coloring(result):
                return [int(c) for c in result]
    
    # Fallback: greedy with 5 colors (should always work for this graph)
    coloring = [0] * n
    for v in range(n):
        used = set()
        for nb in adj[v]:
            if nb < v:
                used.add(coloring[nb])
        for c in range(5):
            if c not in used:
                coloring[v] = c
                break
    
    if is_valid_coloring(coloring):
        return [int(c) for c in coloring]
    
    # Ultimate fallback: distinct colors
    return list(range(n))
