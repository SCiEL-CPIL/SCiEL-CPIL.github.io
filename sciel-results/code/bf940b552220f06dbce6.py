import time
import numpy as np

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    """Random simple k-regular (disconnected) graph via random pairing of stubs."""
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = int(stubs[perm[a]])
        v = int(stubs[perm[a + 1]])
        if u == v or v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

def _rand_connected_regular(n, k, rng):
    """Random SIMPLE k-regular CONNECTED graph via random spanning tree (Wilson)
    plus a k-1 random perfect matching on the tree edges (which keeps the graph
    connected and adds k-1 extra edges per vertex)."""
    if k < 2:
        return None
    # --- random spanning tree (Wilson's / loop-erased self-avoiding walk) ---
    parent = [-1] * n
    in_tree = [False] * n
    in_tree[0] = True
    frontier = [0]
    while frontier:
        u = frontier.pop()
        choices = [x for x in adj[u] if not in_tree[x]]
        if not choices:
            continue
        # loop-erased random walk from u until we hit the tree
        path = [u]
        seen = {u}
        w = u
        while not in_tree[w]:
            w = adj[w][_randrange(rng, k)]
            if w in seen:
                path = path[:path.index(w) + 1]
                seen = set(path)
            else:
                path.append(w)
                seen.add(w)
        v = path[-1]
        if in_tree[v]:
            continue
        parent[v] = path[-2]
        in_tree[v] = True
        frontier.append(v)

    edges = []
    for v in range(1, n):
        edges.append((parent[v], v))

    # --- add k-1 extra edges via a random perfect matching on the tree edges ---
    m = len(edges)  # = n-1
    if k - 1 > m:
        return None
    need = k - 1
    pairs = list(range(m))
    # Fisher-Yates on first need positions
    for i in range(need):
        j = _randrange(rng, m - i)
        pairs[i], pairs[i + j] = pairs[i + j], pairs[i]
    chosen = pairs[:need]

    adj2 = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    def add_edge(a, b):
        if a == b or b in seen[a]:
            return False
        adj2[a].append(b)
        adj2[b].append(a)
        seen[a].add(b)
        seen[b].add(a)
        return True

    for e in edges:
        if not add_edge(e[0], e[1]):
            return None
    for c in chosen:
        if not add_edge(edges[c][0], edges[c][1]):
            return None
    # degree check
    for v in range(n):
        if len(adj2[v]) != k:
            return None
    return adj2

def _girth_ok(adj, g):
    """True iff the (connected) graph has no cycle of length < g."""
    n = len(adj)
    adj_t = [tuple(a) for a in adj]
    for s in range(n):
        dist = {}
        queue = []
        for v in adj_t[s]:
            key = v * n + s
            dist[key] = 1
            queue.append((v, s))
        head = 0
        while head < len(queue):
            u, p = queue[head]
            head += 1
            d = dist[u * n + p]
            if d >= g - 1:
                continue
            for w in adj_t[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                key = w * n + u
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return True

def _luw_fallback(k, g):
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

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        vidx = index[vertex]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vidx, index[other]))
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 120s limit

    # Moore bound (lower bound on n)
    r = (g - 1) // 2
    moore = 1
    for i in range(r):
        moore += k * (k - 1) ** i
    n_start = moore
    if n_start < 2 * k:
        n_start = 2 * k

    rng = 0x9E3779B97F4A7C15
    n = n_start
    while n <= 1300 and time.time() < deadline:
        if n < 450:
            trials = 120
        elif n < 750:
            trials = 60
        elif n < 1050:
            trials = 30
        else:
            trials = 12

        for _ in range(trials):
            if time.time() > deadline:
                break
            adj = _rand_connected_regular(n, k, rng)
            if adj is None:
                continue
            if _girth_ok(adj, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

        n += 1

    # Guaranteed connected fallback: Lazebnik-Ustimenko-Woldar construction
    return _luw_fallback(k, g)
