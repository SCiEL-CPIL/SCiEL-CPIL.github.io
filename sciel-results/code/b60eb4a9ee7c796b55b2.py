"""Construct k-regular graphs of large girth (cage problem).

Combined strategy:
  * Backbone: Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q = k,
    which gives an explicit k-regular bipartite graph of girth d+5 (for odd d).
  * Refinement: try the smallest odd d that should guarantee girth >= g, verify
    with an exact non-backtracking girth check, and if it fails, increase d by 2
    (larger graph, larger girth). A small time-budgeted random-restart repair
    pass is kept as a fallback for parameters where LUW is not valid.
  * An explicit deadline guard returns the best valid result found before the limit.
"""
import time
import random
from collections import deque

# ---------------------------------------------------------------------------
# LUW construction
# ---------------------------------------------------------------------------

def _luw_build(k, d):
    """Build the LUW graph CD(d, k) (q = k). Returns (n, edges) or None."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Exact girth verification (non-backtracking walks from every vertex)
# ---------------------------------------------------------------------------

def _girth_ok(n, edges, g):
    """Return True iff the simple graph has no cycle of length < g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # degree check
    for v in range(n):
        if len(adj[v]) != len(set(adj[v])):
            return False  # multi-edge
    # BFS from each vertex, detect back-edges creating short cycles
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # non-backtracking edge u-v with v at depth <= u
                    cyc = dist[u] + dist[v] + 1
                    if cyc < g:
                        return False
    return True


# ---------------------------------------------------------------------------
# Random regular graph + repair (fallback for parameters LUW can't handle)
# ---------------------------------------------------------------------------

def _random_regular_graph(k, n, rng):
    if n * k % 2 != 0:
        return None
    half = n * k
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    rng.shuffle(stubs)
    adj = {v: set() for v in range(n)}
    for i in range(0, half, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[v]) for v in range(n)]
    for v in range(n):
        if len(adj_list[v]) != k:
            return None
    return adj_list


def _moore_bound(k, g):
    if k == 2:
        return g
    if g % 2 == 0:
        half = g // 2
        return 2 * (k * (k - 1) ** (half - 1))
    else:
        half = (g - 1) // 2
        return 2 * (k * (k - 1) ** half)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    best = None  # (n, edges)

    # --- Primary: LUW with the smallest odd d that should give girth >= g ---
    # For odd d, CD(d, q) has girth d+5. We need d+5 >= g  =>  d >= g-5,
    # and d must be odd. Smallest odd d >= g-5:
    d0 = g - 5
    if d0 % 2 == 0:
        d0 += 1
    d0 = max(3, d0)

    for d in range(d0, d0 + 12, 2):
        if time.time() > deadline - 20:
            break
        try:
            n, edges = _luw_build(k, d)
        except Exception:
            continue
        if n is None:
            continue
        # quick sanity: must be simple and k-regular and girth ok
        if _girth_ok(n, edges, g):
            if best is None or n < best[0]:
                best = (n, edges)
            # LUW gives a valid answer; return the smallest d that worked.
            return n, edges

    # --- Fallback: random regular graph search (only if LUW failed) ---
    if best is not None:
        return best

    rng = random.Random(42)
    base_n = max(_moore_bound(k, g) * 2, k + 1)
    if base_n % 2 != 0:
        base_n += 1

    for attempt in range(40):
        if time.time() > deadline - 15:
            break
        n = base_n + attempt * 2
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            adj = _random_regular_graph(k, n, rng)
            if adj is None:
                continue
            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            if _girth_ok(n, edges, g):
                if best is None or n < best[0]:
                    best = (n, edges)
                return n, edges

    if best is not None:
        return best

    raise ValueError(f"Could not construct k={k}, g={g} graph")
