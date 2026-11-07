"""Rewritten program: greedy maximal-clique extraction for split-graph edge partition."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, edges):
        adj = {}
        for (u, v) in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        cliques = []
        while any(adj.values()):
            # start from the vertex with the largest remaining degree
            start = max((v for v in adj if adj[v]), key=lambda v: len(adj[v]))
            clique = [start]
            cand = set(adj[start])
            while cand:
                best_v = None
                best_deg = -1
                for v in cand:
                    if all(v in adj[c] for c in clique):
                        d = len(adj[v])
                        if d > best_deg:
                            best_deg = d
                            best_v = v
                if best_v is None:
                    break
                clique.append(best_v)
                cand = {v for v in cand if v in adj[best_v]}
            if len(clique) < 2:
                nb = next(iter(adj[start]))
                clique = [start, nb]
            # remove the edges spanned by this clique
            for i in range(len(clique)):
                a = clique[i]
                for j in range(i + 1, len(clique)):
                    b = clique[j]
                    adj[a].discard(b)
                    adj[b].discard(a)
            cliques.append([int(x) for x in clique])
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        result.append(solve(n, k, edges))
    return result
# EVOLVE-BLOCK-END
