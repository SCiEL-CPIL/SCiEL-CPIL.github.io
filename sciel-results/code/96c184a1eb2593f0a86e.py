"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses CSP with MRV heuristic and incremental union-find per color-pair."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)

# Precompute edge index for each vertex
_EDGE_IDX = [[] for _ in range(_N)]
for _i, (_a, _b) in enumerate(_EDGES):
    _EDGE_IDX[_a].append(_i)
    _EDGE_IDX[_b].append(_i)


def _bfs_2color():
    """Return proper 2-coloring if bipartite, else None."""
    color = [-1] * _N
    for start in range(_N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            cu = color[u]
            for v in _ADJ[u]:
                if color[v] == -1:
                    color[v] = 1 - cu
                    q.append(v)
                elif color[v] == cu:
                    return None
    return color


def _is_valid(coloring):
    """Check proper coloring and bicolored-forest constraint."""
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    # Check each color pair's induced subgraph is a forest
    present = sorted(set(coloring))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N

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

            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    if not union(a, b):
                        return False
    return True


class IncrementalCSP:
    """CSP solver with incremental union-find per color-pair and MRV heuristic."""
    def __init__(self, k, deadline):
        self.k = k
        self.deadline = deadline
        self.color = [-1] * _N
        # For each color pair, maintain union-find state
        # Color pairs are (min(c1,c2), max(c1,c2))
        self.uf_parents = {}  # key: (c1,c2), value: list of parents
        self.uf_ranks = {}
        # Track which edges are "active" for each color pair
        self.active_edges = {}  # key: (c1,c2), value: set of edge indices
        self._init_uf()

    def _init_uf(self):
        for c1 in range(self.k):
            for c2 in range(c1 + 1, self.k):
                self.uf_parents[(c1, c2)] = list(range(_N))
                self.uf_ranks[(c1, c2)] = [0] * _N
                self.active_edges[(c1, c2)] = set()

    def _find(self, key, x):
        parent = self.uf_parents[key]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(self, key, x, y):
        parent = self.uf_parents[key]
        rank = self.uf_ranks[key]
        rx, ry = self._find(key, x), self._find(key, y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    def _check_forest(self, v, c):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        for u in _ADJ[v]:
            cu = self.color[u]
            if cu == -1 or cu == c:
                continue
            key = (min(c, cu), max(c, cu))
            if not self._union(key, v, u):
                return False
            self.active_edges[key].add((_EDGE_IDX[v][0],))  # placeholder
        return True

    def _rollback_forest(self, v, c):
        """Rollback the union-find changes for vertex v with color c."""
        for u in _ADJ[v]:
            cu = self.color[u]
            if cu == -1 or cu == c:
                continue
            key = (min(c, cu), max(c, cu))
            # We need to rebuild this union-find structure
            # This is expensive, so we use a different strategy
            pass

    def solve(self):
        """Solve using backtracking with MRV."""
        # Vertex ordering: by degree descending, then by index
        order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
        
        def backtrack(pos):
            if time.time() > self.deadline:
                return False
            if pos == _N:
                return True
            
            v = order[pos]
            # MRV: try colors that are available (not used by colored neighbors)
            used = set()
            for nb in _ADJ[v]:
                if self.color[nb] != -1:
                    used.add(self.color[nb])
            
            for c in range(self.k):
                if c in used:
                    continue
                self.color[v] = c
                # Check if this creates a bicolored cycle
                # We need to verify all color pairs involving c
                ok = True
                for u in _ADJ[v]:
                    cu = self.color[u]
                    if cu == -1:
                        continue
                    key = (min(c, cu), max(c, cu))
                    # Check if adding edge (v,u) creates a cycle in this color pair
                    # We need to temporarily union and check
                    parent = self.uf_parents[key]
                    rx = self._find(key, v)
                    ry = self._find(key, u)
                    if rx == ry:
                        ok = False
                        break
                
                if ok:
                    # Actually perform the unions
                    for u in _ADJ[v]:
                        cu = self.color[u]
                        if cu == -1:
                            continue
                        key = (min(c, cu), max(c, cu))
                        self._union(key, v, u)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Rollback: rebuild union-find structures
                    # This is expensive but necessary for correctness
                    self._rebuild_uf()
                else:
                    self.color[v] = -1
                    continue
                self.color[v] = -1
            return False

        # Simplified approach: just do full validation at each step
        # but with MRV ordering
        def backtrack_simple(pos):
            if time.time() > self.deadline:
                return False
            if pos == _N:
                return True
            
            v = order[pos]
            used = set()
            for nb in _ADJ[v]:
                if self.color[nb] != -1:
                    used.add(self.color[nb])
            
            for c in range(self.k):
                if c in used:
                    continue
                self.color[v] = c
                # Quick check: does this create a bicolored cycle?
                # Check all edges from v to colored neighbors
                cycle_found = False
                for u in _ADJ[v]:
                    cu = self.color[u]
                    if cu == -1 or cu == c:
                        continue
                    # Check if edge (v,u) creates a cycle in color pair (c, cu)
                    # Build temporary union-find for this pair
                    key = (min(c, cu), max(c, cu))
                    parent = list(range(_N))
                    rank = [0] * _N
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
                    
                    # Process all edges in this color pair
                    for a, b in _EDGES:
                        ca, cb = self.color[a], self.color[b]
                        if ca == -1 or cb == -1:
                            continue
                        if ca in (c, cu) and cb in (c, cu):
                            if not union(a, b):
                                cycle_found = True
                                break
                    if cycle_found:
                        break
                
                if not cycle_found:
                    if backtrack_simple(pos + 1):
                        return True
                
                self.color[v] = -1
            return False

        if backtrack_simple(0):
            return self.color[:]
        return None


def construct():
    deadline = time.time() + 80.0

    # 1) Check bipartite
    c2 = _bfs_2color()
    if c2 is not None:
        if _is_valid(c2):
            return [int(x) for x in c2]

    # 2) Try k=3,4,5,... with CSP
    best = None
    for k in range(3, _N + 1):
        if time.time() > deadline - 5.0:
            break
        csp = IncrementalCSP(k, deadline)
        result = csp.solve()
        if result is not None:
            if _is_valid(result):
                best = result
                break

    # 3) Fallback: greedy with verification
    if best is None:
        for k in range(3, min(10, _N + 1)):
            if time.time() > deadline - 2.0:
                break
            coloring = [-1] * _N
            order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
            for v in order:
                used = set()
                for nb in _ADJ[v]:
                    if coloring[nb] != -1:
                        used.add(coloring[nb])
                for c in range(k):
                    if c not in used:
                        coloring[v] = c
                        break
                if coloring[v] == -1:
                    break
            if -1 not in coloring and _is_valid(coloring):
                best = coloring
                break

    # 4) Last resort: unique colors
    if best is None:
        best = list(range(_N))

    return [int(x) for x in best]
