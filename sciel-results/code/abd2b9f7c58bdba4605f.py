"""Rewritten solver for clique (edge-disjoint) partition of three split graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        # cross edges: independent-set vertex v -> set of clique neighbors
        nbrs = [set() for _ in range(n)]
        for v in range(k, n):
            s = nbrs[v]
            for u in range(k):
                if rng.random() < p:
                    s.add(u)
        # clique-internal edges
        Cedges = [(u, v) for u in range(k) for v in range(u + 1, k)]

        used = set()
        cliques = []

        # --- cover cross edges: pack each independent vertex's still-available neighbors ---
        for v in sorted(range(k, n), key=lambda x: (-len(nbrs[x]), x)):
            avail = [u for u in sorted(nbrs[v]) if (u, v) not in used]
            if not avail:
                continue
            S = []
            for u in avail:
                ok = True
                for w in S:
                    if (w, u) in used:
                        ok = False
                        break
                if ok:
                    S.append(u)
            for u in S:
                used.add((u, v))
            if len(S) >= 1:
                cliques.append([v] + S)

        # --- cover remaining clique-internal edges with edge-disjoint cliques ---
        rem = [(a, b) for (a, b) in Cedges if (a, b) not in used]
        if rem:
            deg = defaultdict(int)
            adj = defaultdict(set)
            for a, b in rem:
                adj[a].add(b)
                adj[b].add(a)
                deg[a] += 1
                deg[b] += 1
            while rem:
                v = max(range(k), key=lambda x: (deg[x], -x))
                if deg[v] == 0:
                    break
                S = []
                for u in sorted(adj[v]):
                    if u == v:
                        continue
                    if deg[u] == 0:
                        continue
                    ok = True
                    for w in S:
                        if w in adj[u]:
                            ok = False
                            break
                    if ok:
                        S.append(u)
                for u in S:
                    for w in S:
                        if w > u:
                            used.add((u, w))
                    used.add((min(v, u), max(v, u)))
                if S:
                    cliques.append([v] + S)
                else:
                    u = min(adj[v])
                    used.add((min(v, u), max(v, u)))
                    cliques.append([v, u])
                for u in list(adj[v]):
                    if deg[u] > 0:
                        deg[u] -= 1
                for u in S:
                    for w in adj[u]:
                        if w != v and deg[w] > 0:
                            deg[w] -= 1
                deg[v] = 0
                rem = [(a, b) for (a, b) in rem if (a, b) not in used]

        return cliques

    start = time.time()
    deadline = start + 140.0
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for i, spec in enumerate(specs):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        result.append(solve(n, k, p, seed))
    return result
# EVOLVE-BLOCK-END
