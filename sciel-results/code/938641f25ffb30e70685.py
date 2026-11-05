import time

_DEADLINE = 115.0


def _luw_edges(q, d):
    """Construct LUW CD(d, q) graph."""
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
        r = (c - 5) % 4
        c0 = c - r
        candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = candidates[r]

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

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vi, index[other]))

    return len(order), edges


def _verify(n, edges, k, g):
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False
    for u in range(n):
        dist = {}
        queue = []
        for w in adj[u]:
            dist[(u, w)] = 1
            queue.append((u, w, 1))
        head = 0
        while head < len(queue):
            prev, cur, L = queue[head]
            head += 1
            if L >= g:
                break
            if cur == u and L >= 3:
                return False
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    queue.append((cur, w, L + 1))
    return True


def construct(k, g):
    start_time = time.time()
    
    if k == 4 and g == 10:
        # LUW with d=4: girth >= 2*4+2 = 10, vertices = 2*4^4 = 512
        d = 4
        n, edges = _luw_edges(4, d)
        if _verify(n, edges, k, g):
            return n, edges
    
    # Fallback for other parameters
    if k >= 3:
        d = g - 5
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1
        n, edges = _luw_edges(k, d)
        if _verify(n, edges, k, g):
            return n, edges
    
    return 0, []
