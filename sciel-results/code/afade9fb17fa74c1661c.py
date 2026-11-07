"""Model-and-optimize: parameterize colorings by position-based patterns and search for valid ones."""
import time
import itertools
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
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

def _try_pattern_search(k, adj, n, deadline):
    """Try structured patterns: row-based, column-based, diagonal, etc."""
    # Graph is 5x10 cylindrical grid: vertex (r,c) = r*10+c, edges connect (r,c)-(r,c+1)%10, (r,c)-(r+1,c)%5
    # Try various pattern families
    
    # Pattern 1: color = (r + c) % k
    for k_try in [k]:
        coloring = [((r + c) % k_try) for r in range(5) for c in range(10)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    
    # Pattern 2: color = (r * 2 + c) % k
    coloring = [((r * 2 + c) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 3: color = (r + 2*c) % k
    coloring = [((r + 2 * c) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 4: color = (2*r + c) % k
    coloring = [((2 * r + c) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 5: color = (r + c) // 2 % k (block pattern)
    coloring = [(((r + c) // 2) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 6: row color + column color mod k
    # Try all small linear combinations: (a*r + b*c) % k for a,b in 0..k-1
    for a in range(k):
        for b in range(k):
            if time.time() > deadline:
                return None
            coloring = [((a * r + b * c) % k) for r in range(5) for c in range(10)]
            if _check_coloring(coloring, P['edges']):
                return coloring
    
    # Pattern 7: (a*r + b*c + c) % k - already covered by linear
    # Pattern 8: (r^2 + c) % k
    coloring = [((r * r + c) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 9: (r + c*c) % k
    coloring = [((r + c * c) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 10: (r*c) % k
    coloring = [((r * c) % k) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    # Pattern 11: (r + c) % k but with offset per row
    for offset in range(k):
        coloring = [(((r + offset) + c) % k) for r in range(5) for c in range(10)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    
    # Pattern 12: checkerboard-like with period
    # (r + c) % 2 * 2 + (r // 2 + c // 2) % k
    coloring = [(((r + c) % 2) * 2 + ((r // 2 + c // 2) % k)) for r in range(5) for c in range(10)]
    if _check_coloring(coloring, P['edges']):
        return coloring
    
    return None

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _backtrack_k_colors(k, adj, n, deadline):
    """Backtracking search for valid k-coloring with bicolored forest constraint."""
    # Order by degree descending, then by index
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    pos_to_vertex = order
    vertex_to_pos = [0] * n
    for i, v in enumerate(order):
        vertex_to_pos[v] = i
    
    coloring = [-1] * n
    
    # Precompute for each vertex and color, which neighbors are already colored
    # to speed up the check
    
    def check_new_cyclic(v, c):
        """Check if assigning color c to v creates any bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == c:
                return False
            if cn == -1:
                continue
            # Check if there's a path from v to nb in subgraph of colors {c, cn} not using edge (v,nb)
            visited = {v}
            stack = [v]
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if w in visited:
                        continue
                    if coloring[w] not in (c, cn):
                        continue
                    # Skip the direct edge v-nb
                    if (cur == v and w == nb) or (cur == nb and w == v):
                        continue
                    if w == nb:
                        return False  # Found alternate path -> cycle
                    visited.add(w)
                    stack.append(w)
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        v = pos_to_vertex[pos]
        for c in range(k):
            # Check adjacency
            ok = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            coloring[v] = c
            if check_new_cyclic(v, c):
                if backtrack(pos + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # Try increasing k: first try pattern search (fast), then backtracking
    for k in range(2, 12):
        if time.time() > deadline - 5:
            break
        
        # Try pattern search first (very fast)
        result = _try_pattern_search(k, adj, n, deadline - 5)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
        
        # Try backtracking
        bt_deadline = deadline - 5
        if time.time() > bt_deadline - 2:
            bt_deadline = time.time() + 0.5
        result = _backtrack_k_colors(k, adj, n, bt_deadline)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy with many colors
    coloring = [0] * n
    for v in range(n):
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    return [int(c) for c in coloring]
