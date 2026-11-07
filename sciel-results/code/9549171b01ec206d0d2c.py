"""Optimized search for minimum-color valid coloring of 5x10 cylindrical grid."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Check if coloring is valid: proper + no bicolored cycles."""
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

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _generate_3color_patterns(n=50):
    """Generate diverse 3-coloring patterns for 5x10 grid."""
    patterns = []
    R, C = 5, 10
    
    # Linear patterns: (a*r + b*c + off) % 3
    for a in range(4):
        for b in range(4):
            for off in range(3):
                patterns.append(lambda r, c, a=a, b=b, off=off: (a*r + b*c + off) % 3)
    
    # Non-linear patterns
    patterns.append(lambda r, c: (r + c + (r*c) % 3) % 3)
    patterns.append(lambda r, c: (r + c + (r*c)//3 % 3) % 3)
    patterns.append(lambda r, c: (r*c + r + c) % 3)
    patterns.append(lambda r, c: (r*c - r + c) % 3)
    patterns.append(lambda r, c: (r*c + r - c) % 3)
    patterns.append(lambda r, c: (r*c - r - c) % 3)
    
    # Modulo-based
    patterns.append(lambda r, c: (r % 3 + c % 3) % 3)
    patterns.append(lambda r, c: (r % 3 * c % 3) % 3)
    patterns.append(lambda r, c: ((r % 3) + (c % 3) + (r*c % 3)) % 3)
    
    # Diagonal variants
    patterns.append(lambda r, c: (r + 2*c) % 3)
    patterns.append(lambda r, c: (2*r + c) % 3)
    patterns.append(lambda r, c: (r - c) % 3)
    patterns.append(lambda r, c: (c - r) % 3)
    
    # Shifted patterns
    for shift in range(3):
        patterns.append(lambda r, c, s=shift: ((r + s) % 3 + (c + s) % 3) % 3)
        patterns.append(lambda r, c, s=shift: ((r + s) % 3 * ((c + s) % 3)) % 3)
    
    return patterns

def _try_3color_patterns(adj, n, deadline):
    """Try all generated 3-color patterns."""
    patterns = _generate_3color_patterns(n)
    for pat in patterns:
        if time.time() > deadline:
            break
        coloring = [pat(i // 10, i % 10) for i in range(n)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    return None

def _backtrack_3color(adj, n, deadline):
    """Systematic backtracking for 3-coloring with cycle detection."""
    order = list(range(n))
    # Sort by degree descending for better pruning
    order.sort(key=lambda x: -len(adj[x]))
    
    coloring = [-1] * n
    
    def check_cyclic(v, c):
        """Check if assigning color c to v creates bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1:
                continue
            if cn == c:
                return False
            # Check if there's a path from v to nb in {c, cn} subgraph excluding edge (v,nb)
            visited = {v}
            stack = [v]
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if w in visited:
                        continue
                    cw = coloring[w]
                    if cw != c and cw != cn:
                        continue
                    # Skip the direct edge (v,nb)
                    if (cur == v and w == nb) or (cur == nb and w == v):
                        continue
                    if w == nb:
                        return False
                    visited.add(w)
                    stack.append(w)
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        v = order[pos]
        for c in range(3):
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            coloring[v] = c
            if check_cyclic(v, c):
                if backtrack(pos + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def _try_4color_patterns(adj, n, deadline):
    """Try structured 4-color patterns."""
    patterns = []
    # Linear patterns
    for a in range(5):
        for b in range(5):
            for off in range(4):
                patterns.append(lambda r, c, a=a, b=b, off=off: (a*r + b*c + off) % 4)
    
    # Non-linear
    patterns.append(lambda r, c: (r + c) % 4)
    patterns.append(lambda r, c: (r * c) % 4)
    patterns.append(lambda r, c: (r + 2*c) % 4)
    patterns.append(lambda r, c: (2*r + c) % 4)
    patterns.append(lambda r, c: (r*c + r + c) % 4)
    
    for pat in patterns:
        if time.time() > deadline:
            break
        coloring = [pat(i // 10, i % 10) for i in range(n)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    return None

def _greedy_4color(adj, n):
    """Greedy 4-coloring."""
    coloring = [0] * n
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    coloring = [0] * n
    for v in order:
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        for c in range(4):
            if c not in used:
                coloring[v] = c
                break
    return coloring

def _local_improve(coloring, adj, n, deadline, target_k=4):
    """Local search to reduce colors or fix violations."""
    coloring = coloring[:]
    
    def is_valid(col):
        return _check_coloring(col, P['edges'])
    
    if not is_valid(coloring):
        # Try to fix by reassigning violating vertices
        for _ in range(1000):
            if time.time() > deadline:
                break
            if is_valid(coloring):
                break
            # Find a violation
            fixed = False
            for u, v in P['edges']:
                if coloring[u] == coloring[v]:
                    # Try to change one
                    for new_c in range(target_k):
                        if new_c == coloring[u]:
                            continue
                        old = coloring[u]
                        coloring[u] = new_c
                        if is_valid(coloring):
                            fixed = True
                            break
                        coloring[u] = old
                    if fixed:
                        break
            if not fixed:
                break
    
    # Try to reduce distinct colors
    distinct = sorted(set(coloring))
    while len(distinct) > target_k and time.time() < deadline:
        # Try to merge two colors
        merged = False
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if time.time() > deadline:
                    break
                c1, c2 = distinct[i], distinct[j]
                new_coloring = [c1 if c == c2 else c for c in coloring]
                if is_valid(new_coloring):
                    coloring = new_coloring
                    distinct = sorted(set(coloring))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    if is_valid(coloring):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # Try 3-coloring first
    result = _try_3color_patterns(adj, n, deadline)
    if result is not None:
        if _check_coloring(result, edges):
            return [int(c) for c in result]
    
    if time.time() < deadline - 15:
        result = _backtrack_3color(adj, n, deadline - 15)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Try 4-coloring
    result = _try_4color_patterns(adj, n, deadline)
    if result is not None:
        if _check_coloring(result, edges):
            return [int(c) for c in result]
    
    # Greedy 4-coloring + local improvement
    coloring = _greedy_4color(adj, n)
    result = _local_improve(coloring, adj, n, deadline, 4)
    if result is not None:
        return [int(c) for c in result]
    
    # Fallback: return greedy coloring
    return [int(c) for c in coloring]
