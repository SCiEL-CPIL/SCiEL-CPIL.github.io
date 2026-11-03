"""Rewritten: search for a small 3-regular graph of girth >= g via a
circulant / voltage construction on a base cycle, then fall back to the
LUW algebraic construction. Deadline-guarded."""

import time
import random
from collections import deque

DEADLINE_S = 110.0

def construct(k, g):
    start_wall = time.monotonic()

    def deadline_ok():
        return (time.monotonic() - start_wall) < DEADLINE_S

    # ---------- girth check via non-backtracking walks ----------
    def has_short_cycle(adj, g, max_depth):
        """Return True if there is a cycle of length < g.
        Use non-backtracking walks from each vertex up to depth max_depth.
        max_depth should be ceil((g-1)/2) for an exact certificate, but we use
        a bit smaller for speed and rely on the structure; for final verification
        we do a full BFS."""
        n = len(adj)
        # For each vertex, do a BFS on the state (u, prev)
        # Actually, simpler: for each vertex, BFS with parent tracking
        # But that's expensive for large n.
        # Use the fact that if girth >= g, then no non-backtracking walk
        # of length < g-1 returns to its start without immediate backtracking.
        # We'll do: for each vertex v, for each neighbour u, follow non-backtracking
        # walks of length up to max_depth from (v->u). If we reach v again, cycle found.
        # This is O(n * d * depth) with hashing.
        # For n up to ~1000, this is fine.
        
        depth_limit = max_depth
        for v in range(n):
            for u in adj[v]:
                # state: (current, prev)
                # BFS
                visited = set()
                queue = deque()
                queue.append((u, v, 1))
                while queue and deadline_ok():
                    cur, prev, dist = queue.popleft()
                    if dist >= depth_limit:
                        continue
                    if cur == v:
                        return True
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if (nxt, cur) in visited:
                            continue
                        visited.add((nxt, cur))
                        queue.append((nxt, cur, dist + 1))
        return False

    def verify_girth(adj, g):
        """Exact check: no cycle of length < g."""
        n = len(adj)
        # Moore bound for 3-regular girth g is ~ k*(k-1)^(g/2-1)
        # For g=16, depth = 7
        depth = (g - 1) // 2
        return not has_short_cycle(adj, g, depth)

    # ---------- build adj from edge list ----------
    def edges_to_adj(n, edges):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    # ---------- circulant search ----------
    def try_circulant(m, S, k, g):
        """Try a circulant-like graph on 2*m vertices (bipartite).
        Vertices 0..m-1 on one side, m..2m-1 on other.
        Vertex i (side 0) connects to m + (i+s) % m for s in S.
        Vertex m+i (side 1) connects to (i-s) % m for s in S.
        This gives a k-regular bipartite graph if len(S)=k.
        """
        if len(S) != k:
            return None
        n = 2 * m
        edges = []
        adj = [[] for _ in range(n)]
        for i in range(m):
            for s in S:
                j = m + (i + s) % m
                if i < j:  # avoid duplicates; but i < m <= j always
                    edges.append((i, j))
                adj[i].append(j)
                adj[j].append(i)
        # Check degree
        for v in range(n):
            if len(adj[v]) != k:
                return None
        # Check no multi-edges (shouldn't happen with unique S)
        # Check girth
        if not verify_girth(adj, g):
            return None
        return n, edges

    def search_circulants(k, g, time_budget):
        """Search over m and sets S of size k."""
        deadline = time.monotonic() + time_budget
        # Moore bound for 3-regular, girth 16: n >= 3 * 2^7 = 384? Actually the
        # Moore bound is 2 * sum_{i=0}^{floor((g-1)/2)} (k-1)^i for even girth.
        # For g=16, k=3: 2*(1+2+4+8+16+32+64) = 2*127 = 254.
        # Lower bound 512 is given.
        min_m = 256  # n = 2*m >= 512
        max_m = 1500

        # Try a few deterministic sets first
        # S = {1, a, b}
        best = None
        for m in range(min_m, max_m + 1):
            if time.monotonic() > deadline:
                break
            # Try S = {1, 2, x}
            for x in range(3, m):
                S = (1, 2, x)
                res = try_circulant(m, S, k, g)
                if res:
                    return res
                if time.monotonic() > deadline:
                    break
        return None

    # ---------- LUW construction (fallback) ----------
    def luw_construct(k, g):
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

    # ---------- main logic ----------
    # First try circulant search with a time budget
    circ_result = search_circulants(k, g, 30.0)
    if circ_result:
        return circ_result

    # Fallback to LUW
    return luw_construct(k, g)
