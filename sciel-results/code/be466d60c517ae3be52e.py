"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))

    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
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

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _try_2_coloring(n, edges):
    """
    The diagonal edges (i, i+1) force c(i) = i%2. Try both parities.
    Then verify the bicolored-forest constraint.
    """
    for base in (0, 1):
        colors = [((i + base) % 2) for i in range(n)]
        if _check_bicolored_forest(colors, n, edges):
            return [int(c) for c in colors]
    return None

def _solve_k_coloring(n, edges, k, deadline, rng):
    """General k-coloring solver with backtracking and incremental forest checking."""
    adj = _build_adjacency(n, edges)
    colors = [-1] * n

    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])

    all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(parent, x, y):
        rx, ry = find(parent, x), find(parent, y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    def check_pair_forest(pair, colors, vertex, new_color):
        """Check pair forest assuming vertex has new_color."""
        c1, c2 = pair
        parent = list(range(n))
        for u, v in edges:
            cu = new_color if u == vertex else colors[u]
            cv = new_color if v == vertex else colors[v]
            if cu == -1 or cv == -1 or cu == cv:
                continue
            pkey = (min(cu, cv), max(cu, cv))
            if pkey == pair:
                if not union(parent, u, v):
                    return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        v = order[idx]

        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])

        color_order = list(range(k))
        rng.shuffle(color_order)

        for color in color_order:
            if color in forbidden:
                continue
            colors[v] = color

            valid = True
            for pair in all_pairs:
                c1, c2 = pair
                if color not in pair:
                    continue
                other = c1 if color == c2 else c2
                has_other = any(colors[nb] == other for nb in adj[v] if colors[nb] != -1)
                if not has_other:
                    continue
                if not check_pair_forest(pair, colors, v, color):
                    valid = False
                    break

            if valid:
                if backtrack(idx + 1):
                    return True
            colors[v] = -1
        return False

    if backtrack(0):
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # Safety margin under 90s wall-clock

    # Fast structural 2-coloring: diagonal path forces alternating parity.
    result = _try_2_coloring(n, edges)
    if result is not None:
        return [int(c) for c in result]

    # Fallback general search: 3 colors
    rng = random.Random(42)
    result = _solve_k_coloring(n, edges, 3, deadline, rng)
    if result is not None:
        return [int(c) for c in result]

    # 4 colors
    result = _solve_k_coloring(n, edges, 4, deadline, rng)
    if result is not None:
        return [int(c) for c in result]

    # 5 colors
    result = _solve_k_coloring(n, edges, 5, deadline, rng)
    if result is not None:
        return [int(c) for c in result]

    # Fallback: all different colors
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
