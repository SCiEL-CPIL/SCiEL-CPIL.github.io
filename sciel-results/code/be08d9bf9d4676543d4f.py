"""Bicolored-forest coloring: exploit the cycle+chord structure with smart backtracking."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_coloring(coloring, edges):
    """Verify coloring: adjacent differ, no bicolored cycle."""
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


def _build_adjacency(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _find_coloring_with_k_colors(k, edges, n, deadline, adj):
    """Find valid coloring with k colors using backtracking with reversible UF."""
    
    # Order vertices: start with high-degree, then follow cycle structure
    # The graph is a 50-cycle with chords. Use a smart ordering.
    # Vertices 0-49 form a cycle. Chords connect i to i+7 (mod 50) roughly.
    # Let's use degree-based ordering but break ties by cycle position.
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    # Precompute for each vertex, which neighbors are earlier in the order
    pos_in_order = {v: i for i, v in enumerate(order)}
    earlier_neighbors = [[] for _ in range(n)]
    later_neighbors = [[] for _ in range(n)]
    for v in range(n):
        for nb in adj[v]:
            if pos_in_order[nb] < pos_in_order[v]:
                earlier_neighbors[v].append(nb)
            else:
                later_neighbors[v].append(nb)
    
    coloring = [-1] * n
    
    # Reversible union-find per color pair
    pair_data = {}
    
    def get_pair(c1, c2):
        return (min(c1, c2), max(c1, c2))
    
    def uf_init(pair):
        if pair not in pair_data:
            pair_data[pair] = {'parent': {}, 'rank': {}, 'log': []}
        return pair_data[pair]
    
    def uf_union_with_log(pair, x, y):
        """Union x and y, logging changes for reversal. Returns False if cycle."""
        d = pair_data[pair]
        parent = d['parent']
        rank = d['rank']
        log = d['log']
        
        if x not in parent:
            parent[x] = x
            rank[x] = 0
            log.append(('init', x, None, None))
        if y not in parent:
            parent[y] = y
            rank[y] = 0
            log.append(('init', y, None, None))
        
        # Find roots without path compression
        rx = x
        while parent[rx] != rx:
            rx = parent[rx]
        ry = y
        while parent[ry] != ry:
            ry = parent[ry]
        
        if rx == ry:
            return False
        
        # Union by rank
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        log.append(('union', ry, rx, rank[rx]))
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
            log.append(('rank', rx, rank[rx] - 1, rank[rx]))
        return True
    
    def uf_undo_last(pair, count):
        """Undo the last 'count' operations logged in the pair's log."""
        d = pair_data[pair]
        parent = d['parent']
        rank = d['rank']
        log = d['log']
        
        for _ in range(count):
            op = log.pop()
            if op[0] == 'init':
                _, x, _, _ = op
                del parent[x]
                del rank[x]
            elif op[0] == 'union':
                _, ry, rx, _ = op
                parent[ry] = ry
            elif op[0] == 'rank':
                _, rx, old_rank, _ = op
                rank[rx] = old_rank
    
    best_coloring = [None]
    node_count = [0]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        
        node_count[0] += 1
        if node_count[0] % 10000 == 0 and time.time() > deadline:
            return False
        
        if pos == n:
            best_coloring[0] = coloring[:]
            return True
        
        v = order[pos]
        
        # Determine valid colors: not used by earlier neighbors
        if pos == 0:
            colors_to_try = [0]
        else:
            used_by_earlier = set()
            for nb in earlier_neighbors[v]:
                if coloring[nb] != -1:
                    used_by_earlier.add(coloring[nb])
            colors_to_try = [c for c in range(k) if c not in used_by_earlier]
        
        if not colors_to_try:
            return False
        
        # Try colors in order
        for c in colors_to_try:
            coloring[v] = c
            
            # Try to add edges to UF for earlier neighbors
            pair_ops = {}
            cycle_found = False
            for nb in earlier_neighbors[v]:
                d = coloring[nb]
                if d == c:
                    cycle_found = True
                    break
                pair = get_pair(c, d)
                uf_init(pair)
                ops_before = len(pair_data[pair]['log'])
                if not uf_union_with_log(pair, v, nb):
                    cycle_found = True
                    break
                ops_after = len(pair_data[pair]['log'])
                pair_ops[pair] = pair_ops.get(pair, 0) + (ops_after - ops_before)
            
            if not cycle_found:
                if backtrack(pos + 1):
                    return True
            
            # Undo
            for pair, num_ops in pair_ops.items():
                uf_undo_last(pair, num_ops)
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        return best_coloring[0]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Try small number of colors first
    for k in range(2, 15):
        if time.time() > deadline - 15:
            break
        result = _find_coloring_with_k_colors(k, edges, n, deadline - 10, adj)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy with many colors
    coloring = [0] * n
    for i in range(n):
        used = set()
        for nb in adj[i]:
            used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[i] = c
                break
    
    if _check_coloring(coloring, edges):
        return [int(c) for c in coloring]
    
    return list(range(n))
