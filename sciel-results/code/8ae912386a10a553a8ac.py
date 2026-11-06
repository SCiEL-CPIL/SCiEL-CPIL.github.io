"""Rewritten program: minimum induced matching coloring via exact B&B on conflict graph."""
# EVOLVE-BLOCK-START
import sys

def construct():
    sys.setrecursionlimit(100000)

    def instance(n, d, seed):
        from random import Random
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

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        m = len(edges)
        edge_set = set(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Build conflict graph bitmask
        conflict = [0] * m
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                a, b = edges[j]
                if a == u or a == v or b == u or b == v:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    if (min(u,a), max(u,a)) in edge_set or (min(u,b), max(u,b)) in edge_set or \
                       (min(v,a), max(v,a)) in edge_set or (min(v,b), max(v,b)) in edge_set:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        # Greedy first-fit for initial upper bound
        deg = [bin(conflict[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -deg[i])
        color_of = [-1] * m
        for v in order:
            used = set()
            for j in range(m):
                if (conflict[v] >> j) & 1 and color_of[j] != -1:
                    used.add(color_of[j])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
        upper_bound = max(color_of) + 1

        # B&B to find minimum coloring
        best = [upper_bound]

        def try_color(k):
            color_assign = [-1] * m
            forbidden = [0] * k

            def bt(idx):
                if idx == m:
                    return True
                v = order[idx]
                for c in range(k):
                    if (forbidden[c] >> v) & 1:
                        continue
                    color_assign[v] = c
                    old = forbidden[c]
                    forbidden[c] = old | conflict[v]
                    if bt(idx + 1):
                        return True
                    forbidden[c] = old
                    color_assign[v] = -1
                return False

            if bt(0):
                # Store the solution
                try_color.solution = color_assign[:]
                return True
            return False

        for k in range(upper_bound - 1, 2, -1):
            if try_color(k):
                best[0] = k
                color_of = try_color.solution
                break

        out.append(color_of)

    return out
# EVOLVE-BLOCK-END
