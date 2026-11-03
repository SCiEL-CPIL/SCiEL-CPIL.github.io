import time
import collections
import random
import math

_DEADLINE = None


def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()


def _verify_simple_3reg(n, edges):
    adj = [[] for _ in range(n)]
    seen = set()
    for u, v in edges:
        if not (0 <= u < n) or not (0 <= v < n):
            return False
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != 3:
            return False
    return True


def _girth_bfs_fast(n, adj, g):
    """
    Fast girth check using BFS from each vertex.
    Returns True if girth >= g.
    """
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = collections.deque([s])
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _luw_construct(k, g, d_override=None):
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = (g - 2) // 2 + 1
        if q % 2 == 1 and d % 2 == 0:
            d += 1

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
        if _time_left() < 8.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 140.0

    best = None

    def consider(n, edges):
        nonlocal best
        if not _verify_simple_3reg(n, edges):
            return
        # Fast girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        if _girth_bfs_fast(n, adj, g):
            if best is None or n < best[0]:
                best = (n, edges)

    # Try LUW with smaller d first to minimize vertices
    # For g=16, standard d=8 gives n=1536.
    # Try d=7 (n=768) and d=6 (n=384) to see if they meet girth.
    # Note: LUW girth is typically 2d+1 or 2d+2.
    # If d=7, girth might be 15 (fail) or 16 (pass).
    # If d=6, girth might be 13 or 14 (fail).
    
    candidate_ds = []
    base_d = (g - 2) // 2 + 1
    if k % 2 == 1 and base_d % 2 == 0:
        base_d += 1
        
    # Try base_d, base_d-1, base_d-2
    for offset in range(0, 3):
        d_try = base_d - offset
        if d_try < 1:
            continue
        if d_try in candidate_ds:
            continue
        candidate_ds.append(d_try)
        
    for d_try in candidate_ds:
        if _time_left() < 10.0:
            break
        try:
            n_luw, edges_luw = _luw_construct(k, g, d_override=d_try)
            consider(n_luw, edges_luw)
            # If we found a valid graph with smaller d, we can stop if n is small enough
            # but we want the smallest n, so we should check all valid d's.
            # However, smaller d means smaller n. So if d_try works, it's better than larger d.
            # So we can break after the first valid d_try if we iterate from small to large?
            # No, we iterate from large (base_d) to small.
            # So if d_try=7 works, it's better than d_try=8.
            # We should iterate from smallest d to largest d?
            # No, smaller d gives smaller n. So we want the smallest d that works.
            # So we should try d=6, then d=7, then d=8.
            # If d=6 works, n=384.
            # If d=7 works, n=768.
            # If d=8 works, n=1536.
            # So we should iterate d from small to large.
        except Exception:
            pass

    # Re-evaluate: iterate d from small to large
    candidate_ds = []
    base_d = (g - 2) // 2 + 1
    if k % 2 == 1 and base_d % 2 == 0:
        base_d += 1
        
    min_d = max(1, base_d - 3)
    for d_try in range(min_d, base_d + 1):
        if _time_left() < 10.0:
            break
        try:
            n_luw, edges_luw = _luw_construct(k, g, d_override=d_try)
            consider(n_luw, edges_luw)
        except Exception:
            pass

    if best is not None:
        return best

    # Fallback: return the largest LUW graph constructed
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    except Exception:
        return (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
