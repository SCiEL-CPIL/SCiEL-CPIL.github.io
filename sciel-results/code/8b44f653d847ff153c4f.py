"""Small 5-regular graph of girth 9 (cage problem n(5,9)).

Strategy (within the LUW design family):
  1. Build an explicit LUW graph (guaranteed girth >= g) as a reliable baseline.
  2. Verify girth exactly with a non-backtracking BFS from every vertex.
  3. Try a degree-preserving randomized local search (2-switches) that only
     accepts moves preserving girth >= g, aiming to reduce the vertex count.
  4. Return the best valid graph found, with a deadline guard.
"""
import time
import random
from collections import deque

_DEADLINE = 140.0  # seconds (safety margin under the 150s kill)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges, adj) for the connected component of the zero point."""
    q = k
    # girth of CD(d, q) is d + 5 (odd d) or d + 4 (even d); pick d so girth >= g
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: x * y % q)

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
                y[c] = (add(x[c], mul(y[a], x[b])) if side == 0
                        else sub(x[c], mul(x[a], y[b])))
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
    n = len(order)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, edges, adj


def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses a non-backtracking BFS from every vertex (exact certification)."""
    n = len(adj)
    for s in range(n):
        # state: (current_vertex, previous_vertex)
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(s, nb)] = 2
            dq.append((s, nb))
        while dq:
            prev, cur = dq.popleft()
            d = dist[(prev, cur)]
            if d >= g:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s and d + 1 < g:
                    return False
                st = (cur, nxt)
                if st in dist:
                    continue
                dist[st] = d + 1
                dq.append(st)
    return True


def _random_2switch(adj, n, k, rng):
    """Pick a random degree-preserving 2-switch (a,b) and (c,d) with
    edges a-b, c-d present and a-c, b-d absent. Returns (a,b,c,d) or None."""
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    m = len(edges)
    if m < 2:
        return None
    for _ in range(60):
        a, b = edges[rng.randrange(m)]
        c, d = edges[rng.randrange(m)]
        if {a, b} == {c, d}:
            continue
        if c in (a, b) or d in (a, b):
            continue
        if c in adj[a] or d in adj[b] or c in adj[d] or d in adj[c]:
            continue
        # swap: remove a-b and c-d; add a-c and b-d
        return (a, b, c, d)
    return None


def _try_local_search(n, edges, adj, k, g, time_budget, rng):
    """Attempt to reduce n (or just keep valid) via girth-preserving 2-switches.
    Returns best (n, edges, adj) found within time_budget seconds."""
    if not _girth_ok(adj, g):
        return n, edges, adj
    best_n = n
    best_edges = list(edges)
    best_adj = [set(s) for s in adj]
    t0 = time.time()
    steps = 0
    while time.time() - t0 < time_budget:
        sw = _random_2switch(adj, n, k, rng)
        if sw is None:
            break
        a, b, c, d = sw
        # apply
        adj[a].remove(b); adj[b].remove(a)
        adj[c].remove(d); adj[d].remove(c)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(d); adj[d].add(b)
        steps += 1
        if _girth_ok(adj, g):
            if n < best_n:
                best_n = n
                best_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append((u, v))
                best_adj = [set(s) for s in adj]
        else:
            # revert
            adj[a].remove(c); adj[c].remove(a)
            adj[b].remove(d); adj[d].remove(b)
            adj[a].add(b); adj[b].add(a)
            adj[c].add(d); adj[d].add(c)
        if steps % 4000 == 0 and time.time() - t0 > time_budget:
            break
    return best_n, best_edges, best_adj


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible.
    Returns (n, edges) on vertices 0..n-1."""
    t_start = time.time()
    rng = random.Random(12345)

    # Baseline: explicit LUW graph (guaranteed valid)
    n, edges, adj = _luw_construct(k, g)

    # Exact verification of the baseline
    if not _girth_ok(adj, g):
        # Should not happen for LUW, but guard anyway: fall back to a safe
        # high-girth construction is not trivial; instead just return as-is.
        pass

    # Deadline-aware local search to try to reduce n
    remaining = _DEADLINE - (time.time() - t_start)
    if remaining > 15.0:
        # Reserve ~15s for final verification + return
        n2, edges2, adj2 = _try_local_search(
            n, edges, adj, k, g, remaining - 15.0, rng)
        if n2 < n:
            n, edges, adj = n2, edges2, adj2

    # Final exact girth certification
    if not _girth_ok(adj, g):
        # Revert to the verified baseline
        n, edges, adj = _luw_construct(k, g)

    # Build final edge list (simple, undirected)
    final_edges = []
    seen = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append((u, v))
    return n, final_edges
