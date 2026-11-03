"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
import time
import random
import numpy as np

_DEADLINE = 142.0  # safety margin before the 150 s kill
_T0 = time.time()

def _elapsed():
    return time.time() - _T0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _gq22():
    """Incidence graph of GQ(2,2): 4-regular bipartite, girth 8, 28 vertices."""
    # 35 points of PG(3,2) as 4-bit nonzero vectors
    pts = [tuple(v) for v in np.array(list(range(1, 16)), dtype=int).reshape(-1, 1).repeat(4, axis=1)
           # build all nonzero 4-bit vectors
           ]
    pts = []
    for i in range(1, 16):
        pts.append(tuple((i >> b) & 1 for b in range(4)))
    # hyperbolic quadric Q: x0*x1 = x2*x3
    Q = set(p for p in pts if p[0] * p[1] == p[2] * p[3])
    points = [p for p in pts if p not in Q]  # 15 points
    # lines of GQ: point sets = {p in points: p is collinear-in-Q with a base pair}
    # A GQ(2,2) line = 3 points of PG(3,2)\Q lying on a line that meets Q in 2 points.
    # Build lines by: for each pair of Q-points (a line of Q), the third PG point on that line
    # is not in Q; group the 15 non-Q points into 5 lines per Q-line... simpler:
    # Use the known fact: lines of GQ(2,2) are the 35-15=... use incidence via polarity.
    # Direct construction: two points p,q of GQ are on a line iff the PG(3,2) line pq meets Q in 0 points
    # (i.e. both p,q off Q and the third point r = p+q (mod2) is also off Q).
    # So a GQ line = {p, q, p^q} all off Q. Enumerate triples.
    lines = []
    seen = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p, q = points[i], points[j]
            r = tuple((p[b] ^ q[b]) for b in range(4))
            if r in set(points) and r not in (p, q):
                tri = tuple(sorted([p, q, r]))
                if tri not in seen:
                    seen.add(tri)
                    lines.append(list(tri))
    # Now build incidence graph
    adj = [[] for _ in range(len(points))]          # point -> lines
    line_pts = [[] for _ in range(len(lines))]      # line -> points
    for li, ln in enumerate(lines):
        for p in ln:
            pi = points.index(p)
            adj[pi].append(li)
            line_pts[li].append(pi)
    n = len(points) + len(lines)
    edges = []
    for pi in range(len(points)):
        for li in adj[pi]:
            edges.append((pi, len(points) + li))
    return n, edges

def _greedy_cage(k, g, target_n, seed, deadline):
    """Grow a k-regular graph with girth >= g via greedy edge addition.
    Returns (n, edges) or None if it cannot complete within deadline."""
    rng = random.Random(seed)
    n = target_n
    adj = [set() for _ in range(n)]
    edges = []
    # girth constraint: no cycle shorter than g. We enforce by BFS-depth check
    # from each endpoint up to depth (g-2)//2 to avoid creating a short cycle.
    half = (g - 2) // 2  # max BFS depth when adding edge (u,v): dist in graph w/o edge must be >= g-1

    def too_close(u, v, depth):
        # BFS from u up to `depth` levels; return True if v reached
        if u == v:
            return True
        seen = {u: 0}
        frontier = [u]
        for d in range(1, depth + 1):
            nxt = []
            for x in frontier:
                for y in adj[x]:
                    if y == v:
                        return True
                    if y not in seen:
                        seen[y] = d
                        nxt.append(y)
            frontier = nxt
        return False

    def candidate_pairs():
        # vertices with degree < k, prefer lowest degree
        cand = [v for v in range(n) if len(adj[v]) < k]
        return cand

    # order of operations: repeatedly pick a low-degree pair and add an edge if safe
    # We do a deterministic-ish greedy with random tie-breaking.
    attempts = 0
    max_attempts = 200000
    while any(len(adj[v]) < k for v in range(n)) and attempts < max_attempts:
        if _elapsed() > deadline:
            return None
        attempts += 1
        # pick two vertices with degree < k, not already adjacent
        low = sorted([v for v in range(n) if len(adj[v]) < k], key=lambda v: (len(adj[v]), v))
        if len(low) < 2:
            return None
        # try to add an edge between two low-degree vertices
        progressed = False
        # shuffle a window of the lowest-degree vertices
        window = low[:min(len(low), 24)]
        rng.shuffle(window)
        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                u, v = window[i], window[j]
                if v in adj[u]:
                    continue
                if too_close(u, v, half):
                    continue
                # add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            # stuck: this seed/target cannot complete
            return None
    if any(len(adj[v]) != k for v in range(n)):
        return None
    return n, edges

def _verify(n, edges, k, g):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    # girth check via BFS from each vertex, no-backtracking
    for s in range(n):
        # BFS state (node, prev), distance
        from collections import deque
        dq = deque()
        dq.append((s, -1, 1))
        visited = {(s, -1)}
        while dq:
            node, prev, dist = dq.popleft()
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == s:
                    if dist < g:
                        return False
                    continue
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    dq.append((nb, node, dist + 1))
    return True

def construct(k, g):
    global _T0
    _T0 = time.time()
    # Known best for k=4, g=10 is 320 (Exoo et al.). Try to build a valid 4-regular
    # girth-10 graph with as few vertices as possible.
    if k == 4 and g == 10:
        # Attempt greedy cage construction targeting 320 vertices.
        deadline = _T0 + 120.0
        for target in [320]:
            for seed in range(0, 200):
                if _elapsed() > 115.0:
                    break
                res = _greedy_cage(4, 10, target, seed, deadline)
                if res is not None:
                    n, edges = res
                    if _verify(n, edges, 4, 10):
                        return n, edges
        # Fallback: GQ(2,2) incidence graph (girth 8) -- only valid if g <= 8.
        # For g=10 this is NOT valid, so we must return something valid.
        # Use a larger safe construction: LUW with sufficient d to guarantee girth >= g.
        return _luw_safe(k, g)
    # General fallback: LUW construction (valid, girth >= g for proper d).
    return _luw_safe(k, g)

def _luw_safe(k, g):
    """Lazebnik-Ustimenko-Woldar construction, valid for prime k (or k=4)."""
    q = k
    # need d such that girth >= d + 5 (odd d) => d >= g - 5
    d = g - 5
    if d < 3:
        d = 3
    if (d % 2) == 0:
        d += 1  # ensure odd for the girth>=d+5 guarantee
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x ^ y
        sub = lambda x,y: x ^ y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x + y) % q
        sub = lambda x,y: (x - y) % q
        mul = lambda x,y: x * y % q
    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
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
