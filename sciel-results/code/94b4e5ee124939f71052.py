"""Direct approach: exploit graph structure for minimal bicolored-forest coloring.
Use incremental union-find per color pair for fast cycle detection during backtracking."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_coloring(coloring, edges):
    """Validate: adjacent differ, and each color-pair subgraph is a forest."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Union-find per color pair
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


class UnionFind:
    __slots__ = ('parent', 'rank', 'n')
    def __init__(self, n=0):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        r = self.rank
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True
    def clone(self):
        uf = UnionFind(self.n)
        uf.parent = self.parent[:]
        uf.rank = self.rank[:]
        return uf


def _find_coloring(k, edges, n, deadline):
    """Backtracking with incremental per-pair union-find."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Vertex ordering: Ladder structure order for better propagation
    # Pairs (i, i+25)
    order = []
    for i in range(25):
        order.append(i)
        if i + 25 < n:
            order.append(i + 25)
    # Ensure all nodes are included
    if len(order) < n:
        for i in range(n):
            if i not in order:
                order.append(i)
    
    coloring = [-1] * n
    # pair_uf: maps (c1, c2) -> UnionFind for that color pair
    pair_uf = {}
    
    def get_pair_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in pair_uf:
            pair_uf[key] = UnionFind(n)
        return pair_uf[key]
    
    def assign(v, c):
        """Assign color c to vertex v, updating pair union-finds. Returns True if valid."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1:
                continue
            if cn == c:
                return False
            key = (min(cn, c), max(cn, c))
            if key not in pair_uf:
                pair_uf[key] = UnionFind()
            uf = pair_uf[key]
            if not uf.union(v, nb):
                return False
        coloring[v] = c
        return True
    
    def unassign(v, c):
        """Undo color c from vertex v, reverting pair union-finds."""
        coloring[v] = -1
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1:
                continue
            if cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            uf = pair_uf[key]
            # Revert: we need to remove the union. Since union-find doesn't support undo easily,
            # we'll just recompute. Actually, for backtracking, we save/restore the UF state.
            # This is expensive. Alternative: don't use incremental UF, use full check at each step.
            pass
    
    # Given the complexity of undoing union-find, let's use a different strategy:
    # At each backtracking step, after tentative assignment, check if any bicolored cycle
    # was created by doing a targeted check: for each neighbor with assigned color, 
    # check if the new edge creates a cycle in that pair's subgraph.
    
    # We'll maintain pair_uf and for undo, we'll save the state of affected UFs.
    
    # Actually, let's use a simpler but effective approach:
    # Try all colorings via backtracking, and at each complete assignment, do full validation.
    # But prune early using adjacency constraints.
    # For the bicolored cycle check, we'll do it incrementally with a "dirty" flag.
    
    # Better approach: since k is small (2-6), the number of color pairs is at most C(k,2)=15.
    # We can maintain per-pair UF and for undo, we'll snapshot the parent/rank dicts for affected pairs.
    
    def backtrack(pos, deadline):
        if time.time() > deadline:
            return None
        if pos == n:
            # Final validation
            if _check_coloring(coloring, edges):
                return coloring[:]
            return None
        
        v = order[pos]
        
        # Try each color
        for c in range(k):
            # Check adjacency: no neighbor with color c
            conflict = False
            for nb in adj[v]:
                if coloring[nb] == c:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Save state of affected pair UFs
            affected_pairs = []
            for nb in adj[v]:
                cn = coloring[nb]
                if cn != -1 and cn != c:
                    key = (min(cn, c), max(cn, c))
                    if key not in affected_pairs:
                        affected_pairs.append(key)
            
            # Snapshot affected UFs
            snapshots = {}
            for key in affected_pairs:
                if key in pair_uf:
                    snapshots[key] = (pair_uf[key].parent[:], pair_uf[key].rank[:])
            
            # Try assignment
            valid = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                key = (min(cn, c), max(cn, c))
                if key not in pair_uf:
                    pair_uf[key] = UnionFind(n)
                uf = pair_uf[key]
                if not uf.union(v, nb):
                    valid = False
                    break
            
            if valid:
                coloring[v] = c
                result = backtrack(pos + 1, deadline)
                if result is not None:
                    return result
                coloring[v] = -1
            
            # Restore state
            for key in affected_pairs:
                if key in snapshots:
                    pair_uf[key].parent = snapshots[key][0]
                    pair_uf[key].rank = snapshots[key][1]
                elif key in pair_uf:
                    # UF was created during this step, remove it if empty
                    if not pair_uf[key].parent:
                        del pair_uf[key]
            
            if time.time() > deadline:
                return None
        
        return None
    
    result = backtrack(0, deadline)
    return result


def construct():
    start = time.time()
    deadline = start + 80  # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try k = 2, 3, 4, ...
    for k in range(2, 12):
        if time.time() > deadline - 15:
            break
        result = _find_coloring(k, edges, n, deadline)
        if result is not None:
            # Verify it's a proper list of Python ints
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: use n colors (always valid)
    return list(range(n))
