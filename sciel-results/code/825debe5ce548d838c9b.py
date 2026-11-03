"""Improved Bipartite Random Greedy with Fast Distance Tracking."""
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Returns (n, edges).
    """
    deadline = time.time() + 115  # Safety margin for 120s limit

    best = None
    try:
        lzw = _lzw_construct(k, g)
        if lzw is not None:
            best = lzw
    except Exception:
        best = None

    if time.time() < deadline:
        base = _moore_bound(k, g)
        cap = best[0] if best else 2000

        # Generate candidate n values starting from Moore bound
        candidates = []
        n = max(base + 2, 430)
        while n <= cap:
            candidates.append(n)
            n = int(n * 1.05) + 2

        # Add specific strategic points
        for extra in [500, 600, 700, 800, 900, 1000, 1100, 1200]:
            if extra <= cap and extra not in candidates:
                candidates.append(extra)

        candidates = sorted(set(candidates))

        for n in candidates:
            if time.time() > deadline - 10:
                break
            res = _bipartite_greedy_fast(k, g, n, deadline)
            if res is not None:
                if best is None or res[0] < best[0]:
                    best = res
                break  # First valid n is smallest for this strategy

    if best is not None:
        return best

    return _lzw_construct(k, g)

def _moore_bound(k, g):
    if g % 2 == 0:
        d = g // 2
        n = k * (k - 1) ** (d - 1)
    else:
        d = (g - 1) // 2
        n = 1 + k * sum((k - 1) ** i for i in range(d))
    return n

def _bipartite_greedy_fast(k, g, n, deadline):
    """
    Bipartite random greedy with fast distance tracking via BFS frontier.
    """
    n_half = n // 2
    if 2 * n_half != n:
        n = 2 * n_half
        n_half = n // 2

    # For bipartite graph, min cycle length = 2 * (dist + 1) >= g
    # So dist >= ceil(g/2) - 1
    min_dist = -(-g // 2) - 1
    max_bfs_depth = min_dist - 1  # We check if dist <= max_bfs_depth

    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            return None

        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges_added = 0
        total_edges_needed = n * k // 2

        # Pre-compute: for each vertex, maintain its BFS frontier up to max_bfs_depth
        # This is expensive, so we'll do lazy updates

        success = True
        while edges_added < total_edges_needed:
            if time.time() > deadline - 3:
                success = False
                break

            # Pick random u from left side with deg < k
            u = None
            for _ in range(50):
                idx = random.randint(0, n_half - 1)
                if deg[idx] < k:
                    u = idx
                    break
            if u is None:
                success = False
                break

            # Pick random v from right side with deg < k
            v = None
            for _ in range(50):
                idx = random.randint(0, n_half - 1)
                cand = n_half + idx
                if deg[cand] < k:
                    v = cand
                    break
            if v is None:
                success = False
                break

            # Check if adding (u, v) creates a short cycle
            # This happens if dist(u, v) in current graph <= max_bfs_depth
            if _too_close(adj, u, v, max_bfs_depth):
                # Try to find alternative v
                found = False
                for _ in range(20):
                    idx = random.randint(0, n_half - 1)
                    cand_v = n_half + idx
                    if deg[cand_v] < k and cand_v != v:
                        if not _too_close(adj, u, cand_v, max_bfs_depth):
                            v = cand_v
                            found = True
                            break
                if not found:
                    # Try alternative u
                    found2 = False
                    for _ in range(20):
                        idx = random.randint(0, n_half - 1)
                        cand_u = idx
                        if deg[cand_u] < k and cand_u != u:
                            if not _too_close(adj, cand_u, v, max_bfs_depth):
                                u = cand_u
                                found2 = True
                                break
                    if not found2:
                        success = False
                        break
            else:
                # Valid pair
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges_added += 1

        if not success:
            continue

        if any(d != k for d in deg):
            continue

        edges = []
        for u in range(n_half):
            for v in adj[u]:
                edges.append((u, v))

        if _verify_girth_fast(n, edges, k, g):
            return (n, edges)

    return None

def _too_close(adj, u, v, limit):
    """
    Returns True if distance between u and v is <= limit.
    Uses BFS with early termination.
    """
    if u == v:
        return True
    if limit <= 0:
        return False

    # BFS from u, stop when we reach depth > limit or find v
    dist = {u: 0}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= limit:
            continue
        for nb in adj[node]:
            if nb == v:
                return True
            if nb not in dist:
                dist[nb] = d + 1
                if d + 1 <= limit:
                    queue.append(nb)
    return False

def _verify_girth_fast(n, edges, k, g):
    """
    Verify k-regular and girth >= g using non-backtracking walks.
    """
    if len(edges) != n * k // 2:
        return False
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    max_depth = (g - 1) // 2

    for start in range(n):
        # Non-backtracking BFS: track (node, prev) pairs
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            visited.add((start, nb))

        while queue:
            node, prev, d = queue.popleft()
            if d >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue  # Non-backtracking
                if nb == start:
                    cycle_len = d + 1
                    if cycle_len < g:
                        return False
                    continue
                if (node, nb) not in visited:
                    visited.add((node, nb))
                    queue.append((node, nb, d + 1))
    return True

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
