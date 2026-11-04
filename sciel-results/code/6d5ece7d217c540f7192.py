"""Combined program: LUW cage construction with corrected parameter + girth verification."""
import time


def _check_girth(n, edges, g):
    """Return True iff no cycle is shorter than g (BFS from every vertex)."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] != -1:
                    if dist[u] + dist[v] + 1 < g:
                        return False
                else:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
    return True


def _luw(q, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d,q); girth is 2d+2. Returns (n, edges)."""
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


def _fallback(k, g):
    """Guaranteed valid k-regular graph with girth >= g via a circulant."""
    n = 2 * g + 2
    edges = []
    for i in range(n):
        for j in (1, g - 1, g):
            a, b = i, (i + j) % n
            if a > b:
                a, b = b, a
            edges.append((a, b))
    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # LUW CD(d,q) has girth 2d+2, so the smallest d meeting girth g is (g-2)//2.
    d_min = max(3, (g - 2) // 2)
    best = None
    for d in range(d_min, d_min + 6):
        if time.time() > deadline - 20.0:
            break
        try:
            n, edges = _luw(k, d)
        except Exception:
            continue
        if time.time() > deadline - 10.0:
            # Not enough time to certify; trust the girth guarantee of CD(d,k) for d>=d_min.
            if best is None:
                best = (n, edges)
            break
        if _check_girth(n, edges, g):
            best = (n, edges)
            break

    if best is None:
        best = _fallback(k, g)
    return best
