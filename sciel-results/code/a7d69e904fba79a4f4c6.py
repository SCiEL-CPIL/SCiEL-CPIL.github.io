"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def solve_with_k_colors(k, max_backtracks=10000000):
        degree = [len(adj[v]) for v in range(n)]
        order = sorted(range(n), key=lambda v: -degree[v])

        uf = {}

        def uf_find(c1, c2, x):
            key = (min(c1, c2), max(c1, c2))
            if key not in uf:
                return x
            parent = uf[key]
            if x not in parent:
                return x
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def uf_union(c1, c2, x, y):
            key = (min(c1, c2), max(c1, c2))
            if key not in uf:
                uf[key] = {}
            parent = uf[key]
            if x not in parent:
                parent[x] = x
            if y not in parent:
                parent[y] = y
            rx = uf_find(c1, c2, x)
            ry = uf_find(c1, c2, y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        def uf_undo(c1, c2, x, y):
            key = (min(c1, c2), max(c1, c2))
            if key not in uf:
                return
            parent = uf[key]
            rx = uf_find(c1, c2, x)
            ry = uf_find(c1, c2, y)
            if rx != ry:
                parent[rx] = rx

        color = [-1] * n
        bt_count = [0]

        def bt(pos):
            if bt_count[0] > max_backtracks or time.time() > deadline:
                return False
            if pos == n:
                return True

            v = order[pos]
            for c in range(k):
                ok = True
                for u in adj[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue

                cycle = False
                undone = []
                for u in adj[v]:
                    d = color[u]
                    if d == -1 or d == c:
                        continue
                    if uf_find(c, d, v) == uf_find(c, d, u):
                        cycle = True
                        break
                    if uf_union(c, d, v, u):
                        undone.append((c, d, v, u))
                    else:
                        cycle = True
                        break

                if cycle:
                    for (c2, d2, v2, u2) in reversed(undone):
                        uf_undo(c2, d2, v2, u2)
                    continue

                color[v] = c
                if bt(pos + 1):
                    return True
                color[v] = -1
                for (c2, d2, v2, u2) in reversed(undone):
                    uf_undo(c2, d2, v2, u2)
                bt_count[0] += 1

            return False

        if bt(0):
            return color
        return None

    best = None
    for k in range(3, 12):
        if time.time() > deadline:
            break
        res = solve_with_k_colors(k)
        if res is not None:
            best = res
            break

    if best is not None:
        uc = sorted(set(best))
        mp = {c: i for i, c in enumerate(uc)}
        return [int(mp[c]) for c in best]

    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
