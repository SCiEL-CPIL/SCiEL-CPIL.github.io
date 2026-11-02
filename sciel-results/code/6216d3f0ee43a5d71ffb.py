"""Construct small k-regular graphs of high girth.
Base: LUW CD(d,q) construction (q=k), girth >= d+5 for odd d.
Refinement: fast girth check (single-root BFS for the shortest cycle) so we
can actually verify and select the smallest valid d within the time budget.
"""
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Smallest odd d with d + 5 >= g.
    min_d = g - 5
    if min_d % 2 == 0:
        min_d += 1
    min_d = max(3, min_d)

    best = None
    for d in range(min_d, min_d + 4):
        if time.time() > deadline - 20:
            break
        n, edges = _luw_construct(k, d)
        if n <= 0 or len(edges) != n * k // 2:
            continue
        if not _verify_girth_fast(n, edges, k, g):
            continue
        if best is None or n < best[0]:
            best = (n, edges)
        if n <= _moore_bound(k, g):
            break

    if best is None:
        n, edges = _luw_construct(k, min_d)
        best = (n, edges)
    return best

def _moore_bound(k, g):
    if g % 2 == 0:
        return 2 * (k - 1) ** (g // 2)
    else:
        return k * (k - 1) ** ((g - 1) // 2 - 1)

def _luw_construct(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _shortest_cycle_len(n, edges):
    adj = _build_adj(n, edges)
    best = n + 1
    for root in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[root] = 0
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            if du >= best // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cl = dist[u] + dist[v] + 1
                    if cl < best:
                        best = cl
    return best

def _verify_girth_fast(n, edges, k, g):
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    if _shortest_cycle_len(n, edges) < g:
        return False
    return True

# Compatibility shims (original names/signatures retained, unused).
def _known_1116_cage():
    return 0, []

def _verify_girth(n, edges, g):
    # original signature had no k; infer k from degrees.
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    k = deg[0] if n else 0
    return _verify_girth_fast(n, edges, k, g)

def _construct_5_9(deadline):
    pass

def _random_5_regular(n, rng):
    pass

def _get_adjacency(edges, n):
    return _build_adj(n, edges)

def _find_short_cycle(adj, n, max_len):
    pass

def _improve_girth(edges, n, target_girth, deadline, rng):
    pass

def _estimate_girth(adj, n, upper_bound):
    pass

def _has_cycle_of_length(adj, n, length):
    pass
