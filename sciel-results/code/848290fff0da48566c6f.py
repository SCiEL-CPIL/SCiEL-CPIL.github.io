"""Find minimal number of colors such that each pair of colors induces a forest.
Uses backtracking with reversible union-find for incremental bicolored cycle detection."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_coloring(coloring, edges):
    """Check if coloring is valid: adjacent differ, and no bicolored cycle."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
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
        for u, v in elist:
            if not union(u, v):
                return False
    return True


class ReversibleUnionFind:
    """Union-Find with undo capability for backtracking."""
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
        """Union x and y. Returns False if they were already connected (cycle)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Record state for undo
        self.undo_stack.append((rx, ry, self.parent[ry], self.rank[rx], self.rank[ry]))
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
    
    def undo(self):
        """Undo the last union operation."""
        if not self.undo_stack:
            return
        rx, ry, old_parent_ry, old_rank_rx, old_rank_ry = self.undo_stack.pop()
        self.parent[ry] = old_parent_ry
        self.rank[rx] = old_rank_rx
        self.rank[ry] = old_rank_ry
    
    def snapshot(self):
        """Return current stack size for later rollback."""
        return len(self.undo_stack)
    
    def rollback(self, snap):
        """Rollback to a previous snapshot."""
        while len(self.undo_stack) > snap:
            self.undo()


def _find_coloring_with_k_colors(k, edges, n, deadline):
    """Try to find a valid coloring with k colors using backtracking with reversible UF."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices: higher degree first for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    coloring = [-1] * n
    # For each pair of colors, maintain a ReversibleUnionFind
    pair_ufs = {}
    
    def get_uf(c1, c2):
        pair = (min(c1, c2), max(c1, c2))
        if pair not in pair_ufs:
            pair_ufs[pair] = ReversibleUnionFind(n)
        return pair_ufs[pair]
    
    # Track snapshots for each UF to allow rollback
    uf_snapshots = {}  # pair -> snapshot
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns list of (uf, snapshot) to rollback, or None if invalid."""
        # Check adjacency constraint
        for nb in adj[v]:
            if coloring[nb] == c:
                return None
        
        coloring[v] = c
        snapshots = []
        
        # Check bicolored cycles for each neighbor with a different color
        for nb in adj[v]:
            if coloring[nb] == -1:
                continue
            d = coloring[nb]
            if d == c:
                continue
            uf = get_uf(c, d)
            pair_key = (min(c, d), max(c, d))
            snap = uf.snapshot()
            if not uf.union(v, nb):
                # Cycle detected - undo this assignment
                coloring[v] = -1
                # Rollback all UFs we modified
                for ufk, sn in snapshots:
                    ufk.rollback(sn)
                return None
            snapshots.append((uf, snap))
        
        return snapshots
    
    def unassign(v, c, snapshots):
        """Undo assignment of color c to vertex v, rolling back UF changes."""
        coloring[v] = -1
        for uf, snap in snapshots:
            uf.rollback(snap)
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        v = order[pos]
        
        # Try colors in order
        for c in range(k):
            # Check if color c is valid for v (adjacency)
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            snapshots = try_assign(v, c)
            if snapshots is not None:
                if backtrack(pos + 1):
                    return True
                unassign(v, c, snapshots)
        
        return False
    
    if backtrack(0):
        return coloring[:]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try k = 2, 3, 4, ... until we find a valid coloring
    best_result = None
    for k in range(2, 12):
        if time.time() > deadline - 10:
            break
        result = _find_coloring_with_k_colors(k, edges, n, deadline)
        if result is not None:
            # Verify
            if _check_coloring(result, edges):
                best_result = [int(c) for c in result]
                break  # Found minimal k, no need to try larger
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy with many colors
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    coloring = [0] * n
    valid = True
    for v in order:
        used_colors = set()
        for nb in adj[v]:
            used_colors.add(coloring[nb])
        
        for c in range(50):
            if c in used_colors:
                continue
            # Check bicolored cycle
            coloring[v] = c
            ok = True
            for nb in adj[v]:
                d = coloring[nb]
                if d == c:
                    ok = False
                    break
                # BFS in {c, d} subgraph
                visited = set()
                queue = [v]
                visited.add(v)
                found_cycle = False
                while queue:
                    cur = queue.pop(0)
                    for w in adj[cur]:
                        if w in visited:
                            continue
                        if coloring[w] in (c, d):
                            if (cur == v and w == nb) or (cur == nb and w == v):
                                continue
                            if w == nb:
                                found_cycle = True
                                break
                            visited.add(w)
                            queue.append(w)
                    if found_cycle:
                        break
                if found_cycle:
                    ok = False
                    break
            if ok:
                break
            else:
                coloring[v] = 0
                continue
        else:
            valid = False
            break
    
    if valid and _check_coloring(coloring, edges):
        return [int(c) for c in coloring]
    
    # Ultimate fallback: 50 distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
