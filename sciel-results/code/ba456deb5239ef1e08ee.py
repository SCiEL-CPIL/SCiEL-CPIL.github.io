"""Rewritten program: fast bicolored-forest coloring via incremental union-find with rollback."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Degeneracy ordering (peeling) for fast backtracking
    order = _degeneracy_order(n, adj)

    deadline = time.time() + 85  # safety margin under 90s

    # Try increasing color counts; this graph should be solvable with 3 or 4.
    for k in range(2, 20):
        if time.time() > deadline - 5:
            break
        res = _solve(n, adj, order, k, deadline)
        if res is not None:
            return res

    # Fallback: greedy with a larger palette
    res = _greedy_fallback(n, adj, order)
    if res is not None:
        return res
    # Last resort (always valid: all distinct)
    return [int(i) for i in range(n)]


def _degeneracy_order(n, adj):
    deg = [len(adj[i]) for i in range(n)]
    remaining = set(range(n))
    order = []
    while remaining:
        v = min(remaining, key=lambda x: (deg[x], x))
        order.append(v)
        remaining.discard(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    return order


def _solve(n, adj, order, k, deadline):
    """Backtracking with proper incremental union-find and rollback."""
    color = [-1] * n
    color_count = [0] * k
    # For each color pair (c1,c2), maintain a union-find structure
    # uf[key] = (parent_list, rank_list, edge_count, log_stack)
    uf = {}

    class UF:
        __slots__ = ['parent', 'rank', 'edges', 'log']
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.edges = 0
            self.log = []  # stack of (child_root, parent_root, rank_before) for rollback

        def find(self, x):
            r = x
            while self.parent[r] != r:
                r = self.parent[r]
            # Path compression (no rollback needed for find since we store roots)
            while x != r:
                self.parent[x], x = r, self.parent[x]
            return r

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False  # cycle detected
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.log.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            self.edges += 1
            return True

        def rollback_to(self, size):
            while len(self.log) > size:
                ry, rx, rank_before = self.log.pop()
                self.parent[ry] = ry
                self.rank[rx] = rank_before
                self.edges -= 1

    def get_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in uf:
            uf[key] = UF(n)
        return uf[key]

    def check_pair(v, c, cu):
        """Check if assigning color c to v creates a cycle in pair (c, cu).
        Returns (ok, uf_obj, log_size_before) if ok, else (False, None, None)."""
        ufo = get_uf(c, cu)
        log_size = len(ufo.log)
        ok = True
        for u in adj[v]:
            cuu = color[u]
            if cuu == c or cuu == cu:
                if not ufo.union(v, u):
                    ok = False
                    break
        if not ok:
            ufo.rollback_to(log_size)
            return False, None, None
        return True, ufo, log_size

    def backtrack(idx):
        if time.time() > deadline - 2:
            return None
        if idx == n:
            return list(color)
        v = order[idx]
        
        # Try colors in order of least used first for better diversity
        colors_to_try = sorted(range(k), key=lambda c: (color_count[c], c))
        
        for c in colors_to_try:
            # Quick adjacency check
            conflict = False
            for u in adj[v]:
                if color[u] == c:
                    conflict = True
                    break
            if conflict:
                continue

            # Tentatively assign
            color[v] = c
            color_count[c] += 1

            # Check all pairs involving c that could have cycles
            # A pair (c, cu) can only get a new edge if v has color c and some neighbor has color cu
            # OR if v has color cu and some neighbor has color c
            # Since v now has color c, we need to check pairs (c, cu) for all cu in {0..k-1}
            # where cu != c and there's at least one vertex with color cu adjacent to v,
            # OR where there's already an edge in the (c,cu) subgraph involving a neighbor of v with color cu
            
            affected_cus = set()
            for u in adj[v]:
                if color[u] != -1 and color[u] != c:
                    affected_cus.add(color[u])
            
            # Also need to check pairs where v connects to another vertex with color c
            # (i.e., pair (c, cu) where some neighbor has color c) - but neighbors can't have color c
            # due to adjacency constraint, so this is already handled.
            
            # However, we also need to check if adding v (color c) creates issues in pairs (c, cu)
            # where cu is any color that has vertices, because v might connect to vertices of color cu
            # through the union-find. Actually, the union-find for pair (c,cu) only includes edges
            # between vertices colored c or cu. So we only need to check pairs where v has a neighbor
            # of color cu (cu != c), which is what affected_cus captures.
            
            # But wait: we also need to check pairs (c, cu) where cu != c even if no neighbor has color cu,
            # because... actually no. If no neighbor of v has color cu, then adding v (color c) doesn't add
            # any new edge to the (c, cu) subgraph. So we only need affected_cus.
            
            # Actually, I realize there's a subtlety: the union-find for pair (c,cu) tracks ALL edges
            # between vertices of color c and vertices of color cu. When we assign v color c, we add
            # edges from v to all its neighbors that have color cu. So yes, only affected_cus matters.

            ok = True
            rollbacks = []  # list of (uf_obj, log_size_before)
            for cu in affected_cus:
                res, ufo, log_size = check_pair(v, c, cu)
                if not res:
                    ok = False
                    break
                rollbacks.append((ufo, log_size))

            if ok:
                result = backtrack(idx + 1)
                if result is not None:
                    return result

            # Rollback
            for ufo, log_size in rollbacks:
                ufo.rollback_to(log_size)
            color[v] = -1
            color_count[c] -= 1

        return None

    return backtrack(0)


def _greedy_fallback(n, adj, order):
    color = [-1] * n
    for v in order:
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(50):
            if c not in used:
                color[v] = c
                break
    if -1 in color:
        return None
    return [int(c) for c in color]


# EVOLVE-BLOCK-END
