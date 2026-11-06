"""Strong edge colouring: randomized greedy restarts + local compression."""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        m = len(edges)
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)
        conf = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in v2e[u]:
                if j != i:
                    conf[i].add(j)
            for j in v2e[v]:
                if j != i:
                    conf[i].add(j)
            for nb in adj[u]:
                for j in v2e.get(nb, ()):
                    conf[i].add(j)
            for nb in adj[v]:
                for j in v2e.get(nb, ()):
                    conf[i].add(j)
        return conf

    def greedy_order(edges, conf, deg, rng):
        m = len(edges)
        # weight[i] = max conflict degree of neighbours + own degree, with random tiebreak
        w = [0.0] * m
        for i in range(m):
            mx = 0
            for j in conf[i]:
                if len(conf[j]) > mx:
                    mx = len(conf[j])
            w[i] = mx + 0.1 * deg[edges[i][0]] + 0.1 * deg[edges[i][1]] + rng.random() * 0.5
        order = sorted(range(m), key=lambda i: w[i], reverse=True)
        return order

    def greedy(edges, conf, order, rng):
        m = len(edges)
        col = [0] * m
        used = [set() for _ in range(m)]  # used[i] = colours present in conf[i]
        for i in order:
            c = 0
            while c in used[i]:
                c += 1
            col[i] = c
            for j in conf[i]:
                used[j].add(c)
        return col

    def compress(col, conf, edges, rng):
        m = len(edges)
        # try to recolor each edge to a smaller colour
        for _ in range(2):
            for i in range(m):
                cur = col[i]
                if cur == 0:
                    continue
                # find colours used by neighbours
                used = set()
                for j in conf[i]:
                    used.add(col[j])
                for c in range(cur):
                    if c not in used:
                        col[i] = c
                        break
        # relabel to 0..k-1
        present = sorted(set(col))
        remap = {c: k for k, c in enumerate(present)}
        return [remap[c] for c in col], len(present)

    def solve_instance(edges, adj, deg, rng):
        m = len(edges)
        if m == 0:
            return []
        conf = build_conflict(edges, adj)
        best_col = None
        best_k = m
        while time.time() < DEADLINE:
            order = greedy_order(edges, conf, deg, rng)
            col = greedy(edges, conf, order, rng)
            col, k = compress(col, conf, edges, rng)
            if k < best_k:
                best_k = k
                best_col = list(col)
            if best_k <= 1:
                break
        return best_col

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        adj = {}
        deg = [0] * n
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            deg[u] += 1
            deg[v] += 1
        rng = Random(seed * 1000 + 7)
        col = solve_instance(edges, adj, deg, rng)
        out.append(col)

    return out
