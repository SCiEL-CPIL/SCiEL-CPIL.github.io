"""Construct a k-regular graph of girth at least g on as few vertices as possible."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque

def _moore_bound(k, g):
    """Lower bound on number of vertices for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: 1 + k * sum_{i=0}^{g/2-2} (k-1)^i
        r = g // 2 - 1
        return 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        # Odd girth: 2 * sum_{i=0}^{(g-1)/2-1} (k-1)^i  ... actually:
        # n >= 2 * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i  -- let me use the standard formula
        # For odd girth g: n >= 2 * (1 + (k-1) + ... + (k-1)^{floor((g-1)/2)-1})
        # More precisely: n >= 2 * sum_{i=0}^{(g-1)//2 - 1} (k-1)^i  when (g-1)//2 >= 1
        # Standard: n >= 2 * ( (k-1)^{g/2} - 1 ) / (k-2) for even... let me just compute directly
        r = (g - 1) // 2
        return 2 * sum((k - 1) ** i for i in range(r))


def _bfs_girth(adj):
    """Compute the girth (shortest cycle) of an undirected graph via BFS from each vertex.
    adj: list of lists of neighbors. Returns girth (int) or large number if no cycle.
    Uses the standard technique: for each vertex, BFS and track parent to detect back-edges.
    For efficiency, we do BFS from each vertex but stop when depth >= current best girth/2.
    """
    n = len(adj)
    best = float('inf')
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] * 2 >= best:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle: dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best:
                        best = cycle_len
    return best if best != float('inf') else 2 * n + 1


def _verify_graph(n, edges, k, g):
    """Verify: simple, k-regular, girth >= g."""
    # Check edge list
    edge_set = set()
    deg = [0] * n
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        deg[i] += 1
        deg[j] += 1
    for d in deg:
        if d != k:
            return False
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    # Check girth
    girth = _bfs_girth(adj)
    return girth >= g


def _lazebnik_ustimenko_woldar(k, g):
    """Construct LZW graph CD(d, q) with q = k.
    This is a q-regular bipartite graph with girth at least d+5 (odd d) or d+4 (even d).
    We pick d so that the girth bound meets or exceeds g.
    Returns (n, edges) for the connected component containing the zero vertex.
    """
    q = k
    # We need girth >= g. The LZW graph CD(d,q) has girth >= 2*floor(d/2)+3 or similar.
    # From the seed: girth is at least d+5 for odd d.
    # Let's pick d = g - 5 if that gives odd d, else g - 4.
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    if d < 3:
        d = 3

    # Define field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define rule[c] = (a, b) for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # BFS from the zero vertex
    # Vertex representation: (side, x) where side in {0,1}, x is tuple of length d+1
    # x[0] is unused (None), x[1]..x[d] are field elements
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for idx, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _greedy_regular_graph(k, g, max_vertices=None):
    """Try to construct a k-regular graph of girth >= g using a greedy/randomized approach.
    This is a fallback; may not succeed for large g."""
    if max_vertices is None:
        max_vertices = 2 * k * k * k  # heuristic upper bound
    # For small cases, try random k-regular graphs and check girth
    # This is very slow for large k, g, so only use for small cases
    if k > 6 or g > 7:
        return None

    for attempt in range(500):
        # Generate a random k-regular graph on n vertices
        # Use the configuration model
        n = max_vertices
        # Actually, let's try smaller n first
        n_try = _moore_bound(k, g)
        if n_try % 2 != 0:
            n_try += 1
        if n_try * k % 2 != 0:
            n_try += 1

        # Configuration model
        half_edges = []
        for v in range(n_try):
            for _ in range(k):
                half_edges.append(v)
        random.shuffle(half_edges)
        edges = []
        seen = set()
        valid = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen:
                valid = False
                break
            seen.add(key)
            edges.append(key)
        if not valid:
            continue
        # Check girth
        adj = [[] for _ in range(n_try)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        girth = _bfs_girth(adj)
        if girth >= g:
            return n_try, edges
    return None


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Try the LZW construction first
    best_n = float('inf')
    best_edges = None

    # LZW graph
    try:
        n_lzw, edges_lzw = _lazebnik_ustimenko_woldar(k, g)
        if n_lzw < best_n:
            # Verify
            if _verify_graph(n_lzw, edges_lzw, k, g):
                best_n = n_lzw
                best_edges = edges_lzw
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        if best_edges is not None:
            return best_n, best_edges
        # Fallback: return something valid
        return _lazebnik_ustimenko_woldar(k, g)

    # Try to improve: for small k, g, try randomized search for smaller graphs
    if k <= 4 and g <= 7:
        try:
            result = _greedy_regular_graph(k, g)
            if result is not None:
                n_g, edges_g = result
                if n_g < best_n:
                    if _verify_graph(n_g, edges_g, k, g):
                        best_n = n_g
                        best_edges = edges_g
        except Exception:
            pass

    if best_edges is None:
        # Should not happen, but fallback
        best_n, best_edges = _lazebnik_ustimenko_woldar(k, g)

    return best_n, best_edges
# EVOLVE-BLOCK-END
