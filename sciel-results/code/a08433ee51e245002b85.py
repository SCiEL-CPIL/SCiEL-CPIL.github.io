import time
import random

def construct(k, g):
    deadline = time.time() + 145.0
    random.seed(42)

    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)

    def find_all_cycles(adj, n, max_len):
        cycles = set()
        for start in range(n):
            def dfs(node, prev, path):
                if len(path) >= max_len:
                    return
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and len(path) >= 3:
                        cycle = tuple(path)
                        min_rot = min(cycle[i:] + cycle[:i] for i in range(len(cycle)))
                        cycles.add(min_rot)
                    elif nb not in path:
                        dfs(nb, node, path + [nb])
            dfs(start, -1, [start])
        return list(cycles)

    cycles = find_all_cycles(adj, n_base, g)

    for m in range(90, 120):
        if time.time() > deadline:
            break
        n = n_base * m
        if n < 448:
            continue
        for _ in range(300):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(len(petersen_edges))]
            ok = True
            for cycle in cycles:
                total = 0
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    idx = None
                    for j, (a, b) in enumerate(petersen_edges):
                        if (a == u and b == v) or (a == v and b == u):
                            idx = j
                            break
                    if idx is not None:
                        if (u, v) == petersen_edges[idx]:
                            total = (total + voltages[idx]) % m
                        else:
                            total = (total - voltages[idx]) % m
                if total == 0:
                    ok = False
                    break
            if ok:
                cover_edges = []
                for (u, v) in petersen_edges:
                    idx = petersen_edges.index((u, v))
                    vol = voltages[idx]
                    for i in range(m):
                        u2 = u * m + i
                        v2 = v * m + ((i + vol) % m)
                        if u2 < v2:
                            cover_edges.append((u2, v2))
                        else:
                            cover_edges.append((v2, u2))
                return n, cover_edges

    # Fallback: L-U-W
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
