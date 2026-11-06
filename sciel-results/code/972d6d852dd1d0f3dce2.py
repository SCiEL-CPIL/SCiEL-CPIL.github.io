import time
from random import Random


def construct():
    t0 = time.time()
    LIMIT = 110.0  # safety margin under the 120 s wall-clock kill

    def instance(n, d, seed):
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

    def conflict_graph(edges):
        eset = set(edges)
        m = len(edges)
        adj = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, d = edges[j]
                if a == c or a == d or b == c or b == d:
                    adj[i].add(j); adj[j].add(i)
                    continue
                if (a, c) in eset or (a, d) in eset or (b, c) in eset or (b, d) in eset:
                    adj[i].add(j); adj[j].add(i)
        return [list(s) for s in adj]

    def dsatur(adj, deg, m, rng):
        color = [-1] * m
        sat = [set() for _ in range(m)]
        uncolored = list(range(m))
        while uncolored:
            scored = [(len(sat[v]), deg[v], v) for v in uncolored]
            scored.sort(reverse=True)
            top = scored[:min(6, len(scored))]
            v = rng.choice([t[2] for t in top])
            used = sat[v]
            c = 0
            while c in used:
                c += 1
            color[v] = c
            uncolored.remove(v)
            for u in adj[v]:
                if color[u] == -1:
                    sat[u].add(c)
        return color

    def reduce_colors(color, adj, m):
        improved = True
        while improved:
            improved = False
            K = max(color)
            vk = [v for v in range(m) if color[v] == K]
            moved = 0
            for v in vk:
                used = {color[u] for u in adj[v]}
                for c in range(K):
                    if c not in used:
                        color[v] = c
                        moved += 1
                        break
            if vk and moved == len(vk):
                improved = True
        return color

    def backtrack_target(adj, deg, m, target, budget):
        """Try to colour with exactly `target` colours via DSATUR backtracking.
        Returns a valid colour list on success, else None."""
        color = [-1] * m
        sat = [set() for _ in range(m)]
        uncolored = list(range(m))
        nodes = [0]

        def pick():
            best = None
            bestkey = None
            for v in uncolored:
                key = (len(sat[v]), deg[v])
                if best is None or key > bestkey:
                    bestkey = key
                    best = v
            return best

        def bt():
            nodes[0] += 1
            if nodes[0] > budget:
                return False
            if not uncolored:
                return True
            v = pick()
            uncolored.remove(v)
            used = sat[v]
            for c in range(target):
                if c in used:
                    continue
                color[v] = c
                touched = []
                for u in adj[v]:
                    if color[u] == -1:
                        sat[u].add(c)
                        touched.append(u)
                if bt():
                    return True
                for u in touched:
                    sat[u].discard(c)
                color[v] = -1
                uncolored.append(v)
            return False

        if bt():
            return color
        return None

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
        adj = conflict_graph(edges)
        deg = [len(a) for a in adj]

        best_k = None
        best_color = None
        rng = Random(987654321)
        nrestart = 600
        for r in range(nrestart):
            if time.time() > LIMIT - 10:
                break
            col = dsatur(adj, deg, m, rng)
            k = max(col) + 1
            if best_k is None or k < best_k:
                best_k = k
                best_color = col[:]

        if best_color is None:
            col = dsatur(adj, deg, m, rng)
            best_color = col
            best_k = max(col) + 1

        # local colour reduction on the best solution
        best_color = reduce_colors(best_color, adj, m)
        best_k = max(best_color) + 1

        # bounded backtracking to try to prove a smaller count feasible
        for target in range(best_k - 1, 0, -1):
            if time.time() > LIMIT - 10:
                break
            sol = backtrack_target(adj, deg, m, target, 300000)
            if sol is not None:
                best_color = sol
                best_k = target
                best_color = reduce_colors(best_color, adj, m)
                best_k = max(best_color) + 1
                break

        out.append(best_color)

    return out
