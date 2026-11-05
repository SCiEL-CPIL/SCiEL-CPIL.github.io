"""
5-regular cage graph (girth 9) construction.
Strategy:
  1. Try randomized local search (simulated annealing with local improvement)
     to find a small 5-regular graph with girth >= 9.
  2. If not found within time, fall back to Lazebnik-Ustimenko-Woldar construction.
"""
import time
import random
import numpy as np

DEADLINE_FRACTION = 0.90  # use 90% of the 150s wall-clock

def _girth_ok(adj, g, max_check=8):
    """
    Check if graph has girth >= g using non-backtracking BFS.
    adj: list of lists of neighbors (0-indexed).
    Returns True if no cycle of length < g is found.
    """
    n = len(adj)
    # For each vertex, do non-backtracking BFS up to depth g-1.
    # If we return to a vertex already seen at depth < g, girth < g.
    for start in range(n):
        # State: (vertex, prev_vertex, depth)
        # We track the shortest non-backtracking path from start to each (vertex, prev) state.
        visited = {}  # (v, prev) -> depth
        queue = [(start, -1, 0)]
        while queue:
            v, prev, d = queue.pop(0)
            if (v, prev) in visited:
                continue
            visited[(v, prev)] = d
            if d >= g:
                continue
            for u in adj[v]:
                if u == prev:
                    continue
                if u == start and d + 1 < g:
                    # Found a cycle of length d+1 < g
                    return False
                if (u, v) not in visited:
                    queue.append((u, v, d + 1))
    return True


def _is_valid_5_regular_girth9(n, edges, g=9):
    """
    Validate: simple graph, 5-regular, girth >= g.
    """
    adj = [[] for _ in range(n)]
    seen_edges = set()
    for i, j in edges:
        if i == j:
            return False
        if i >= n or j >= n:
            return False
        key = (min(i, j), max(i, j))
        if key in seen_edges:
            return False
        seen_edges.add(key)
        adj[i].append(j)
        adj[j].append(i)
    for i in range(n):
        if len(adj[i]) != 5:
            return False
    # Check girth
    return _girth_ok(adj, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION

    # Moore bound for (5,9): 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
    moore_bound = 1
    for i in range(1, g // 2 + 1):
        moore_bound += k * (k - 1) ** (i - 1)
    # Actually for girth 9 (odd), Moore bound = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # = 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # So n must be >= 426.

    # Try random search for small n
    best_n, best_edges = None, None
    rng = random.Random(42)

    # Try a few random constructions for small n
    # For 5-regular graphs, n must be even (since 5*n must be even)
    # Start from n = 426 (even? 426 is even, yes)
    
    # Actually, let's try a smarter approach: use the LUW graph as a starting point
    # and try to reduce its size via edge swaps if possible.
    # But that's complex. Instead, let's try a few random 5-regular graphs.
    
    # For now, just return the LUW construction if we don't find something better.
    # The LUW construction for k=5, g=9 gives a specific n.
    
    luw_n, luw_edges = _luw_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Try to find a smaller graph via random search
    # n must be >= 426 and even
    for n in range(426, min(luw_n, 500), 2):
        if time.time() > deadline:
            break
        # Try a few random 5-regular graphs
        for attempt in range(10):
            if time.time() > deadline:
                break
            # Generate random 5-regular graph
            # Use configuration model: create 5*n half-edges and randomly pair them
            try:
                edges = _random_5_regular(n, rng)
                if edges is not None:
                    if _is_valid_5_regular_girth9(n, edges, g):
                        return n, edges
            except Exception:
                pass
    
    return best_n, best_edges


def _random_5_regular(n, rng):
    """
    Generate a random 5-regular graph on n vertices using configuration model.
    Returns list of edges or None if multigraph/self-loop detected.
    """
    if n < 5 or n % 2 != 0:
        return None
    # Create list of half-edges
    half_edges = []
    for v in range(n):
        for _ in range(5):
            half_edges.append(v)
    rng.shuffle(half_edges)
    # Pair them up
    adj = [[] for _ in range(n)]
    seen = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in seen:
            return None
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    # Verify 5-regular
    for i in range(n):
        if len(adj[i]) != 5:
            return None
    edges = list(seen)
    return edges
