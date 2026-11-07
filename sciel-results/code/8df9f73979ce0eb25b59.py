import time
import numpy as np
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check(coloring):
    """Validate: adjacent differ and every two-color induced subgraph is a forest."""
    adj = _build_adj()
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))

    for (a, b), edges in pair_edges.items():
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        vlist = list(verts)
        idx = {v: i for i, v in enumerate(vlist)}
        par = list(range(len(vlist)))
        rn = [0] * len(vlist)

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def un(x, y):
            rx, ry = f(x), f(y)
            if rx == ry:
                return False
            if rn[rx] < rn[ry]:
                rx, ry = ry, rx
            par[ry] = rx
            if rn[rx] == rn[ry]:
                rn[rx] += 1
            return True

        for u, v in edges:
            if not un(idx[u], idx[v]):
                return False
    return True


def _backtrack_with_mrv(adj, nc, deadline, init_coloring=None):
    """
    Backtracking search with MRV heuristic and efficient bicolored-cycle detection.
    Uses incremental union-find per color pair to detect cycles.
    """
    n = P['n']
    coloring = [-1] * n
    if init_coloring is not None:
        for i, c in enumerate(init_coloring):
            if 0 <= c < nc:
                coloring[i] = c
            else:
                coloring[i] = -1

    # For each color pair (a, b), maintain a union-find to track the forest
    # We'll track: for each pair, which vertices are in that pair's subgraph
    # and whether adding an edge would create a cycle.

    # Precompute: for each vertex, its neighbors
    # We'll use a dictionary: pair -> {vertex: component_id} via union-find

    class UF:
        def __init__(self, n):
            self.par = list(range(n))
            self.rank = [0] * n

        def find(self, x):
            while self.par[x] != x:
                self.par[x] = self.par[self.par[x]]
                x = self.par[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.par[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True

    # pair_ufs: (a,b) -> UF instance
    # pair_verts: (a,b) -> set of vertices in that subgraph
    # pair_edge_count: (a,b) -> number of edges added
    # We need to be careful: the UF should be over all n vertices, but we only care about
    # vertices that actually have color a or b.

    pair_ufs = {}
    pair_vert_sets = {}
    pair_edge_counts = {}

    def get_pair_key(c1, c2):
        return (min(c1, c2), max(c1, c2))

    def add_vertex_to_pairs(v, c):
        """When vertex v gets color c, it may be part of color pairs with its neighbors' colors."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = get_pair_key(c, cn)
            if key not in pair_ufs:
                pair_ufs[key] = UF(n)
                pair_vert_sets[key] = set()
                pair_edge_counts[key] = 0
            pair_vert_sets[key].add(v)

    def add_edge_to_pairs(u, c1, v, c2):
        """When edge (u,v) is colored with colors c1,c2, check if it creates a cycle."""
        if c1 == c2:
            return False  # invalid
        key = get_pair_key(c1, c2)
        if key not in pair_ufs:
            return False
        uf = pair_ufs[key]
        if not uf.union(u, v):
            return False  # cycle detected
        pair_edge_counts[key] += 1
        return True

    def remove_vertex_from_pairs(v, c):
        """Undo: remove vertex v of color c from all pairs."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = get_pair_key(c, cn)
            if key in pair_vert_sets:
                pair_vert_sets[key].discard(v)
            # Note: we don't actually remove edges from the UF; we just rebuild if needed.
            # For simplicity, we'll rebuild the pair's state when removing.

    def remove_edge_from_pairs(u, c1, v, c2):
        """Undo edge addition. This is tricky with union-find (can't easily undo).
        We'll rebuild the pair's UF from scratch when we need to backtrack."""
        pass

    # Given the difficulty of undoing union-find operations, let's use a different approach:
    # Instead of incremental, we'll do a full check after each assignment but optimize it.
    # Actually, let's try a different strategy: use the fact that the graph is small (50 vertices)
    # and just do the full _check at each node, but with MRV to prune early.

    # Revised approach: MRV backtracking with full validation at each step
    def validate_partial(coloring):
        """Check if the partial coloring is still valid (no cycles in any color pair subgraph)."""
        adj_local = adj
        n_local = len(coloring)
        
        # Check adjacent vertices have different colors
        for u in range(n_local):
            for v in adj_local[u]:
                if u < v and coloring[u] != -1 and coloring[v] != -1:
                    if coloring[u] == coloring[v]:
                        return False
        
        # Check for bicolored cycles using union-find on each color pair
        pair_edges = defaultdict(list)
        for u in range(n_local):
            cu = coloring[u]
            if cu == -1:
                continue
            for v in adj_local[u]:
                if u >= v:
                    continue
                cv = coloring[v]
                if cv == -1:
                    continue
                if cu == cv:
                    continue
                key = (min(cu, cv), max(cu, cv))
                pair_edges[key].append((u, v))
        
        for key, edges in pair_edges.items():
            verts = set()
            for u, v in edges:
                verts.add(u)
                verts.add(v)
            if len(verts) == 0:
                continue
            vlist = list(verts)
            idx = {v: i for i, v in enumerate(vlist)}
            par = list(range(len(vlist)))
            rn = [0] * len(vlist)
            
            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            
            for u, v in edges:
                ru, rv = find(idx[u]), find(idx[v])
                if ru == rv:
                    return False
                if rn[ru] < rn[rv]:
                    ru, rv = rv, ru
                par[rv] = ru
                if rn[ru] == rn[rv]:
                    rn[ru] += 1
        return True

    def get_mrv_vertex(coloring):
        """Find uncolored vertex with fewest available colors."""
        min_count = nc + 1
        best_v = -1
        best_colors = []
        
        for v in range(n):
            if coloring[v] == -1:
                used = set()
                for nb in adj[v]:
                    if coloring[nb] != -1:
                        used.add(coloring[nb])
                avail = [c for c in range(nc) if c not in used]
                if len(avail) == 0:
                    return -1, []
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_colors = avail
                    if min_count == 1:
                        break
        
        return best_v, best_colors

    def dfs(depth):
        if time.time() > deadline:
            return None
        
        # Check if all colored
        if depth == n:
            if _check(coloring):
                return coloring[:]
            return None
        
        v, colors = get_mrv_vertex(coloring)
        if v == -1:
            return None
        
        for c in colors:
            coloring[v] = c
            if validate_partial(coloring):
                res = dfs(depth + 1)
                if res is not None:
                    return res
            coloring[v] = -1
        return None

    colored_count = sum(1 for c in coloring if c != -1)
    return dfs(colored_count)


def _try_greedy_with_local_search(adj, nc, deadline):
    """Greedy coloring followed by local search to fix bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
    # Greedy: order by degree descending
    coloring = [-1] * n
    ordering = sorted(range(n), key=lambda x: -len(adj[x]))
    for v in ordering:
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        for c in range(nc):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            return None
    
    if _check(coloring):
        return coloring[:]
    
    # Local search: try to swap colors to fix bicolored cycles
    best_coloring = coloring[:]
    for iteration in range(1000):
        if time.time() > deadline:
            break
        if _check(best_coloring):
            return best_coloring[:]
        
        # Find all bicolored cycles
        adj_local = adj
        pair_edges = defaultdict(list)
        for u in range(n):
            for v in adj_local[u]:
                if u < v:
                    c1, c2 = best_coloring[u], best_coloring[v]
                    if c1 != c2:
                        pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))
        
        found_cycle = False
        for key, edges_in_pair in pair_edges.items():
            verts = set()
            for u, v in edges_in_pair:
                verts.add(u)
                verts.add(v)
            vlist = list(verts)
            idx = {v: i for i, v in enumerate(vlist)}
            par = list(range(len(vlist)))
            rn = [0] * len(vlist)
            
            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            
            cycle_found = False
            for u, v in edges_in_pair:
                ru, rv = find(idx[u]), find(idx[v])
                if ru == rv:
                    # Cycle found! Try to fix by changing one of the vertices' colors
                    for w in [u, v]:
                        for new_c in range(nc):
                            if new_c == best_coloring[w]:
                                continue
                            # Check if new_c is valid for w (not used by neighbors)
                            valid = True
                            for nb in adj_local[w]:
                                if best_coloring[nb] == new_c:
                                    valid = False
                                    break
                            if valid:
                                old_c = best_coloring[w]
                                best_coloring[w] = new_c
                                if _check(best_coloring):
                                    return best_coloring[:]
                                best_coloring[w] = old_c
                                found_cycle = True
                                break
                        if found_cycle:
                            break
                    if found_cycle:
                        break
                else:
                    if rn[ru] < rn[rv]:
                        ru, rv = rv, ru
                    par[rv] = ru
                    if rn[ru] == rn[rv]:
                        rn[ru] += 1
            
            if found_cycle:
                break
        
        if not found_cycle:
            break
    
    if _check(best_coloring):
        return best_coloring[:]
    return None


def _direct_4coloring():
    """
    Direct construction of a 4-coloring for the 5x10 toroidal grid.
    Vertex i is at row i//10, col i%10.
    For a toroidal grid, we can use color = (row + col) % 4, but this may create
    bicolored cycles. Let's try a different pattern.
    
    Actually, for a 5x10 toroidal grid, the chromatic number of the torus T(5,10)
    is known to be related to the girth. Let's try several patterns.
    """
    n = P['n']
    patterns = [
        lambda r, c: (r + c) % 4,
        lambda r, c: (2*r + c) % 4,
        lambda r, c: (r + 2*c) % 4,
        lambda r, c: (r + c + r*c) % 4,
        lambda r, c: (r*2 + c*3) % 4,
        lambda r, c: (r*3 + c*2) % 4,
        lambda r, c: (r + c*2 + r*c) % 4,
        lambda r, c: (r*2 + c + r*c) % 4,
    ]
    
    for pat in patterns:
        coloring = [0] * n
        for v in range(n):
            r = v // 10
            c = v % 10
            coloring[v] = pat(r, c)
        if _check(coloring):
            return coloring
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    adj = _build_adj()
    
    # Strategy 1: Direct construction of 4-coloring
    result = _direct_4coloring()
    if result is not None:
        return [int(x) for x in result]
    
    # Strategy 2: MRV backtracking for 3 colors
    if time.time() < deadline - 30:
        result = _backtrack_with_mrv(adj, 3, deadline - 10)
        if result is not None and _check(result):
            return [int(x) for x in result]
    
    # Strategy 3: MRV backtracking for 4 colors
    if time.time() < deadline - 20:
        result = _backtrack_with_mrv(adj, 4, deadline - 10)
        if result is not None and _check(result):
            return [int(x) for x in result]
    
    # Strategy 4: Greedy with local search for 4 colors
    if time.time() < deadline - 10:
        result = _try_greedy_with_local_search(adj, 4, deadline - 5)
        if result is not None and _check(result):
            return [int(x) for x in result]
    
    # Strategy 5: Greedy with local search for 5 colors
    if time.time() < deadline - 5:
        result = _try_greedy_with_local_search(adj, 5, deadline - 2)
        if result is not None and _check(result):
            return [int(x) for x in result]
    
    # Fallback: all distinct colors
    return [int(i) for i in range(P['n'])]
