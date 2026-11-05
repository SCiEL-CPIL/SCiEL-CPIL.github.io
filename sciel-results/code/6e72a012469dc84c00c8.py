import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def solve(adj):
        n = len(adj)
        # Precompute all triangles
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        triangles.append((u, v, w))

        # All edges
        remaining = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining.add((u, v))

        # Edge -> list of triangle indices
        edge_to_tris = {}
        tri_edges = []
        for i, (u, v, w) in enumerate(triangles):
            te = frozenset([(u, v), (u, w), (v, w)])
            tri_edges.append(te)
            for e in te:
                edge_to_tris.setdefault(e, []).append(i)

        cliques = []
        active = set(range(len(tri_edges)))

        # Greedy: for each remaining edge, pick the active triangle covering
        # the most remaining edges that contains it.
        # To speed up, process in batches.
        while remaining:
            if time.time() > deadline - 3.0:
                break
            e = next(iter(remaining))
            best_tri = None
            best_count = 0
            for ti in edge_to_tris.get(e, []):
                if ti not in active:
                    continue
                te = tri_edges[ti]
                cnt = 0
                for x in te:
                    if x in remaining:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_tri = ti
            if best_tri is not None:
                u, v, w = triangles[best_tri]
                cliques.append([u, v, w])
                for x in tri_edges[best_tri]:
                    remaining.discard(x)
                active.discard(best_tri)
            else:
                a, b = e
                cliques.append([a, b])
                remaining.discard(e)

        # Cover leftovers
        for e in remaining:
            a, b = e
            cliques.append([a, b])

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj = random_graph(n, p, seed)
        cliques = solve(adj)
        results.append(cliques)

    return results
