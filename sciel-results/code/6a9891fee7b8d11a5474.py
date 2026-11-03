import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    if k != 3 or g != 16:
        return 0, []

    # LUW construction: q=2, d=10
    q = 2
    d = 10

    # GF(2) tables
    add_tbl = [[0, 1], [1, 0]]
    mul_tbl = [[0, 0], [0, 1]]

    # Rule table: rule[c] = (a, b) for c in 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    n_per_side = 1 << d
    n_total = 2 * n_per_side

    def idx_to_coords(idx):
        coords = [0] * (d + 1)
        for i in range(d):
            coords[i + 1] = (idx >> i) & 1
        return coords

    def coords_to_idx(coords):
        idx = 0
        for i in range(d):
            idx |= coords[i + 1] << i
        return idx

    def neighbors(side, idx):
        x = idx_to_coords(idx)
        result = []
        other_side = 1 - side
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_tbl[x[c]][mul_tbl[y[a]][x[b]]]
                else:
                    y[c] = add_tbl[x[c]][mul_tbl[x[a]][y[b]]]
            result.append(other_side * n_per_side + coords_to_idx(y))
        return result

    # Build graph
    adj = [[] for _ in range(n_total)]
    edges = []
    visited = [False] * n_total

    from collections import deque as dq
    start = 0
    visited[start] = True
    queue = dq([start])

    while queue:
        v = queue.popleft()
        side = v // n_per_side
        idx = v % n_per_side
        for nb in neighbors(side, idx):
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
            if side == 0:
                adj[v].append(nb)
                adj[nb].append(v)
                edges.append((v, nb))

    n = sum(1 for x in visited if x)

    # Verify girth
    def verify(n, adj, min_g):
        max_d = (min_g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            par = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_d:
                    continue
                pu = par[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        par[v] = u
                        q.append(v)
                    elif dv >= du:
                        if du + dv + 1 < min_g:
                            return False
        return True

    if not verify(n, adj, g):
        return 0, []

    return n, edges
