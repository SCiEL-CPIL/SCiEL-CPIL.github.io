<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _nonback_girth(adj):
    """Exact girth via non-backtracking BFS from every vertex."""
    n = len(adj)
    if n == 0:
        return 999
    best = 999
    for s in range(n):
        # state: (node, prev) -> depth
        dist = {}
        stack = []
        for nb in adj[s]:
            dist[(s, nb)] = 1
            stack.append((s, nb))
        while stack:
            u, v = stack.pop()
            d = dist[(u, v)]
            if d >= best:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                key = (v, w)
                if key in dist:
                    if d + 1 < best:
                        best = d + 1
                else:
                    dist[key] = d + 1
                    stack.append(key)
    return best

def _validate(n, edges, k, g):
    adj = [[] for _ in range(n)]
    seen = set()
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            return False
        seen.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    girth = _nonback_girth(adj)
    return girth >= g

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _try_circulant(k, g, deadline):
    """Try circulant graphs C(n, S) for small n."""
    # Moore bound: n >= 1 + k * sum_{i=0}^{floor((g-2)/2)} (k-1)^i
    from math import floor
    half = (g - 2) // 2
    moore = 1 + k
    for i in range(1, half + 1):
        moore += k * (k - 1) ** i
    # Try a few candidate n values near the Moore bound
    candidates = []
    for n in range(max(moore, k + 1), min(moore + 200, 3000)):
        if n % 2 == 0 and k % 2 == 1:
            continue
        if n % 2 == 1 and k % 2 == 0:
            continue
        candidates.append(n)
    random.shuffle(candidates)
    for n in candidates[:50]:
        if time.time() > deadline:
            break
        # Try random symmetric difference sets
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random k-subset of {1, ..., n//2} with symmetry
            half_n = n // 2
            if k > half_n * 2:
                continue
            S = set()
            while len(S) < k:
                s = random.randint(1, half_n)
                S.add(s)
            edges = []
            adj = [[] for _ in range(n)]
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    if j > i:
                        edges.append((i, j))
                        adj[i].append(j)
                        adj[j].append(i)
            # Check degree
            if all(len(adj[i]) == k for i in range(n)):
                girth = _nonback_girth(adj)
                if girth >= g:
                    return n, edges
    return None

def _try_random_regular(k, g, n_target, deadline):
    """Try to construct a random regular graph with girth >= g."""
    from math import floor
    half = (g - 2) // 2
    moore = 1 + k
    for i in range(1, half + 1):
        moore += k * (k - 1) ** i
    n = max(n_target, moore, k + 1)
    # Ensure n*k is even
    if (n * k) % 2 != 0:
        n += 1
    for attempt in range(30):
        if time.time() > deadline:
            break
        # Configuration model
        if (n * k) % 2 != 0:
            continue
        half_edges = n * k // 2
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        valid = True
        seen_edges = set()
        for idx in range(0, n * k, 2):
            u, v = stubs[idx], stubs[idx + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_edges:
                valid = False
                break
            seen_edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            continue
        if any(len(adj[i]) != k for i in range(n)):
            continue
        girth = _nonback_girth(adj)
        if girth >= g:
            edges = [(min(u, v), max(u, v)) for u, v in seen_edges]
            return n, edges
    return None

def _local_search_reduce(n, edges, k, g, deadline):
    """Try to reduce vertex count via edge swaps that preserve degree and girth."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    best_n = n
    best_edges = list(edges)
    # Simple local search: try removing a vertex and reconnecting
    # This is complex; skip for now and rely on other constructions
    return best_n, best_edges

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Now I need to add the `construct` function that ties everything together:

