"""Fast greedy + incremental local search for bicolored-forest coloring."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _validate(coloring, edges):
    """Check adjacency constraint and bicolored forest constraint via union-find per pair."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair = (cu, cv)
        else:
            pair = (cv, cu)
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


def _pair_cycle_check(coloring, adj, c1, c2):
    """Return True if the subgraph of vertices colored c1 or c2 contains a cycle."""
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    for u in range(len(coloring)):
        if coloring[u] == c1 or coloring[u] == c2:
            for v in adj[u]:
                if v > u and (coloring[v] == c1 or coloring[v] == c2):
                    if not union(u, v):
                        return True
    return False


def _greedy_color(n, adj, k, order, color_func):
    """
    Greedy: assign colors in given order using color_func(v, coloring) -> color or None.
    Uses incremental bicolored-forest check (only checks affected pairs).
    Returns valid coloring or None.
    """
    coloring = [-1] * n
    for v in order:
        best_c = None
        for c in range(k):
            # Adjacency check
            ok = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            # Bicolored forest check: only need to check pairs (c, d) for assigned neighbors
            coloring[v] = c
            cycle_found = False
            for nb in adj[v]:
                d = coloring[nb]
                if d == -1 or d == c:
                    continue
                c1, c2 = (c, d) if c < d else (d, c)
                if _pair_cycle_check(coloring, adj, c1, c2):
                    cycle_found = True
                    break
            coloring[v] = -1
            if not cycle_found:
                best_c = c
                break
        if best_c is None:
            return None
        coloring[v] = best_c
    return coloring


def _local_search_fast(start_coloring, k, n, edges, adj, deadline, max_iter=50000, seed=42):
    """
    Local search with incremental violation tracking.
    Only recomputes bicolored-forest check for pairs affected by a color change.
    """
    coloring = start_coloring[:]
    rng = random.Random(seed)
    start = time.time()

    def full_violations(coloring_local):
        """Count total violations (adjacency + bicolored cycles)."""
        v = 0
        for u, w in edges:
            if coloring_local[u] == coloring_local[w]:
                v += 1
        pair_edges = defaultdict(list)
        for u, w in edges:
            cu, cv = coloring_local[u], coloring_local[w]
            if cu != cv:
                pair = (cu, cv) if cu < cv else (cv, cu)
                pair_edges[pair].append((u, w))
        for pair, elist in pair_edges.items():
            parent = {}
            def find(x):
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent[x], parent[x])
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            for u, w in elist:
                if not union(u, w):
                    v += 1
                    break
        return v

    def delta_violations(v, old_c, new_c, coloring_local):
        """
        Compute change in violations when vertex v changes from old_c to new_c.
        Only checks affected pairs: (old_c, d) and (new_c, d) for neighbors d.
        """
        delta = 0
        # Adjacency: count neighbors with color new_c
        for nb in adj[v]:
            if coloring_local[nb] == new_c:
                delta += 1
        # Bicolored: for each neighbor with color d != new_c and d != old_c,
        # check if (new_c, d) pair gets a cycle, and if (old_c, d) pair loses a cycle
        # Simplified: check if new assignment creates cycles in affected pairs
        affected_pairs = set()
        for nb in adj[v]:
            d = coloring_local[nb]
            if d == new_c or d == old_c or d == -1:
                continue
            affected_pairs.add((min(new_c, d), max(new_c, d)))
            affected_pairs.add((min(old_c, d), max(old_c, d)))

        # Check new pairs for cycles
        for (c1, c2) in affected_pairs:
            if c1 == new_c or c2 == new_c:
                # Temporarily set v to new_c and check
                old_val = coloring_local[v]
                coloring_local[v] = new_c
                if _pair_cycle_check(coloring_local, adj, c1, c2):
                    delta += 1
                coloring_local[v] = old_val

        # Check old pairs: if they had a cycle, removing v's edge might break it
        # This is harder; approximate by checking if the pair still has a cycle without v
        # For simplicity, we'll just do a full check at the end
        return delta

    current_v = full_violations(coloring)
    best_coloring = coloring[:]
    best_v = current_v
    no_improve = 0
    max_no_improve = 2000

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_v == 0:
            return coloring

        v = rng.randint(0, n - 1)
        old_c = coloring[v]

        # Try all colors, pick best
        best_c = old_c
        best_delta = 0
        for c in range(k):
            if c == old_c:
                continue
            # Quick adjacency check
            adj_conflict = False
            for nb in adj[v]:
                if coloring[nb] == c:
                    adj_conflict = True
                    break
            if adj_conflict:
                continue
            # Check bicolored for affected pairs
            coloring[v] = c
            cycle_ok = True
            for nb in adj[v]:
                d = coloring[nb]
                if d == -1 or d == c:
                    continue
                c1, c2 = (c, d) if c < d else (d, c)
                if _pair_cycle_check(coloring, adj, c1, c2):
                    cycle_ok = False
                    break
            coloring[v] = old_c
            if not cycle_ok:
                continue
            # Estimate improvement: if this color reduces conflicts
            # Just accept it (will verify with full check)
            best_c = c
            break  # First valid color is fine

        if best_c != old_c:
            coloring[v] = best_c
            new_v = full_violations(coloring)
            if new_v < best_v:
                best_v = new_v
                best_coloring = coloring[:]
                no_improve = 0
            elif new_v == best_v:
                no_improve += 1
            else:
                # Revert
                coloring[v] = old_c
                no_improve += 1
            current_v = full_violations(coloring)
            if current_v == 0:
                return coloring
        else:
            no_improve += 1

        if no_improve > max_no_improve:
            break

    if _validate(best_coloring, edges):
        return best_coloring
    return None


def _try_grid_patterns(n, k, adj):
    """Try structured grid-based patterns with incremental validation."""
    # The graph is a 5x10 grid with wrap-around in some dimensions
    # Vertex i is at row i//10, col i%10
    # Edges: horizontal (i, i+1 within row), vertical (i, i+10), and some diagonals/wraps
    # Let's try: color = (row * a + col * b) % k for various a, b
    for a in range(1, k + 1):
        for b in range(1, k + 1):
            coloring = []
            for i in range(n):
                row = i // 10
                col = i % 10
                coloring.append((row * a + col * b) % k)
            if _validate(coloring, P['edges']):
                return coloring
    # Also try: color = (row * a + col * b + row*col) % k
    for a in range(1, k + 1):
        for b in range(1, k + 1):
            coloring = []
            for i in range(n):
                row = i // 10
                col = i % 10
                coloring.append((row * a + col * b + row * col) % k)
            if _validate(coloring, P['edges']):
                return coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)

    # Strategy 1: Try structured grid patterns
    for k in [4, 5, 6]:
        if time.time() > deadline - 70:
            break
        result = _try_grid_patterns(n, k, adj)
        if result is not None:
            return [int(c) for c in result]

    # Strategy 2: Greedy with different orderings
    # Try several vertex orderings and color orderings
    orderings = [
        list(range(n)),  # natural order
        sorted(range(n), key=lambda x: -len(adj[x])),  # degree descending
        list(reversed(range(n))),  # reverse natural
    ]
    for k in [4, 5, 6, 7]:
        if time.time() > deadline - 50:
            break
        for order in orderings:
            result = _greedy_color(n, adj, k, order, None)
            if result is not None and _validate(result, edges):
                return [int(c) for c in result]

    # Strategy 3: Local search from multiple starting points
    rng = random.Random(42)
    start_patterns = []
    # Random starts
    for seed in range(15):
        start_patterns.append([random.Random(seed).randint(0, 4) for _ in range(n)])
    # Grid-based starts
    for a in [1, 2, 3]:
        for b in [1, 2, 3]:
            p = []
            for i in range(n):
                row = i // 10
                col = i % 10
                p.append((row * a + col * b) % 5)
            start_patterns.append(p)
    # Checkerboard-like
    p = [(i // 10 + i % 10) % 5 for i in range(n)]
    start_patterns.append(p)
    p = [(i // 10 * 2 + i % 10) % 5 for i in range(n)]
    start_patterns.append(p)

    for k in [5, 4, 6]:
        if time.time() > deadline - 10:
            break
        for pat in start_patterns:
            if time.time() > deadline - 5:
                break
            # Scale pattern to k colors if needed
            if k != 5:
                pat = [c % k for c in pat]
            result = _local_search_fast(pat, k, n, edges, adj, deadline - 5, max_iter=30000, seed=rng.randint(0, 10000))
            if result is not None and _validate(result, edges):
                return [int(c) for c in result]

    # Fallback: proper coloring with 5 colors (always valid for proper coloring,
    # but may have bicolored cycles - still better than 50 colors)
    coloring = [i % 5 for i in range(n)]
    if not _validate(coloring, edges):
        # Try to fix with more colors
        coloring = [i % 7 for i in range(n)]
    if not _validate(coloring, edges):
        coloring = list(range(n))

    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END
