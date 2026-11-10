"""Rewritten: exact 2-coloring via GF(2) cycle-free 2-lifting."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _solve_gf2(A, b):
    """Solve A x = b over GF(2). A is list of row bitmasks, b list of bits.
    Returns (x, feasible) where x is a solution bitmask or None."""
    n = len(b)
    m = max((r.bit_length() for r in A), default=0)
    if m == 0:
        return (0, all(v == 0 for v in b))
    # Build augmented matrix: each row = (coeff_mask << 1) | b_i
    mat = [(A[i] << 1) | b[i] for i in range(n)]
    # Gaussian elimination
    pivot_col = [-1] * n
    rank = 0
    for col in range(m):
        # find pivot
        piv = -1
        for r in range(rank, n):
            if (mat[r] >> col) & 1:
                piv = r
                break
        if piv < 0:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        for r in range(n):
            if r != rank and ((mat[r] >> col) & 1):
                mat[r] ^= mat[rank]
        pivot_col[rank] = col
        rank += 1
        if rank == n:
            break
    # Check consistency: rows that are all-zero in coeff but b=1
    for r in range(rank, n):
        if mat[r] & 1:  # b bit set but no coefficients
            return (None, False)
    # Free variables: set to 0 for a particular solution
    x = 0
    for r in range(rank):
        if (mat[r] >> 1) & 1:  # b bit is 1
            x |= 1 << pivot_col[r]
    return (x, True)

def _verify(colors, edges):
    """Check: proper coloring + no bicolored cycle (via union-find per color pair)."""
    n = len(colors)
    # proper
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # bicolored cycles
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            continue
        adj[(min(cu, cv), max(cu, cv))].append((u, v))
    for pair, elist in adj.items():
        # union-find on vertices for this color pair
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

def _solve_2coloring():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    # Orient each edge: u < v
    oriented = []
    for u, v in edges:
        if u < v:
            oriented.append((u, v))
        else:
            oriented.append((v, u))
    # Variables: one per edge (index 0..m-1)
    # Vertex equations: for each vertex v, XOR of incident edge vars = 0
    # Build incidence rows
    vert_rows = [0] * n
    for ei, (u, v) in enumerate(oriented):
        bit = 1 << ei
        vert_rows[u] ^= bit
        vert_rows[v] ^= bit
    # Cycle equations: find a cycle basis
    # Build adjacency with edge indices
    adj = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(oriented):
        adj[u].append((v, ei))
        adj[v].append((u, ei))
    # Find cycle basis via spanning tree (BFS)
    tree_edges = set()
    parent_v = [-1] * n
    parent_e = [-1] * n
    visited = [False] * n
    from collections import deque
    q = deque([0])
    visited[0] = True
    while q:
        u = q.popleft()
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent_v[v] = u
                parent_e[v] = ei
                tree_edges.add(ei)
                q.append(v)
    # Non-tree edges give fundamental cycles
    cycle_rows = []
    for ei in range(m):
        if ei in tree_edges:
            continue
        u, v = oriented[ei]
        # Build cycle: u -e-> v, then path v->u in tree
        cycle_mask = 1 << ei
        # climb from v to u
        cur = v
        while cur != u:
            pe = parent_e[cur]
            cycle_mask |= 1 << pe
            cur = parent_v[cur]
        cycle_rows.append(cycle_mask)
    # Combine all equations
    A = vert_rows + cycle_rows
    b = [0] * len(A)
    x, feasible = _solve_gf2(A, b)
    if not feasible or x is None:
        return None
    # Compute vertex colors via BFS: color[0]=0, color[v] = color[u] XOR edge_var
    colors = [-1] * n
    colors[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v, ei in adj[u]:
            if colors[v] == -1:
                colors[v] = colors[u] ^ ((x >> ei) & 1)
                q.append(v)
    # Verify
    if _verify(colors, edges):
        return colors
    return None

def _local_search_3color():
    """Try to find a valid 3-coloring via local search (fallback / improvement)."""
    n = P['n']
    edges = P['edges']
    rng = random.Random(12345)
    # Start from 2-coloring if available, else random
    base = _solve_2coloring()
    if base is not None:
        colors = list(base)
    else:
        colors = [rng.randint(0, 2) for _ in range(n)]
    # Precompute adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Objective: number of bicolored cycles (approx by counting edges in 2-cycles via union-find is expensive)
    # Instead, use a penalty: for each color pair, count edges; a bicolored cycle exists iff edges > vertices-1 in that pair's subgraph
    # We'll just do proper-coloring + minimize max color-pair edge count
    def cost(c):
        # proper
        pen = 0
        for u, v in edges:
            if c[u] == c[v]:
                pen += 1000
        # count edges per color pair
        from collections import defaultdict
        cnt = defaultdict(int)
        for u, v in edges:
            cu, cv = c[u], c[v]
            if cu != cv:
                cnt[(min(cu, cv), max(cu, cv))] += 1
        # vertices per color
        vcnt = defaultdict(int)
        for i in range(n):
            vcnt[c[i]] += 1
        # penalty if edges > vertices - 1 (cycle present)
        for (a, b), e in cnt.items():
            va = vcnt[a] + vcnt[b]
            if e > va - 1:
                pen += (e - (va - 1)) * 10
        return pen
    best = colors[:]
    best_c = cost(best)
    deadline = time.time() + 30.0
    T = 10.0
    while time.time() < deadline:
        i = rng.randrange(n)
        old = colors[i]
        new = rng.randint(0, 2)
        if new == old:
            continue
        colors[i] = new
        c = cost(colors)
        if c < best_c or rng.random() < (0.01 / (1 + (c - best_c) / 100)):
            best = colors[:]
            best_c = c
        else:
            colors[i] = old
        T *= 0.999
    # Final verify
    if _verify(best, edges):
        return best
    return None

def construct():
    deadline = time.time() + 50.0
    # Primary: exact 2-coloring
    colors = _solve_2coloring()
    if colors is not None and _verify(colors, P['edges']):
        # distinct labels
        return [int(c) for c in colors]
    # Fallback: local search
    if time.time() < deadline:
        c3 = _local_search_3color()
        if c3 is not None and _verify(c3, P['edges']):
            return [int(c) for c in c3]
    # Last resort: trivial valid (50 colors)
    return [int(i) for i in range(P['n'])]
# EVOLVE-BLOCK-END
