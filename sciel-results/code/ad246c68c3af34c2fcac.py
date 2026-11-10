"""Clique partition of split graphs via edge-assignment to maximal cliques."""
import time
from random import Random


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for u, v in edges:
            if u < k and v < k:
                continue
            if u >= k:
                u, v = v, u
            adj[u].add(v)
            adj[v].add(u)

        S = {}
        for s in range(k, n):
            S[s] = frozenset(v for v in adj[s] if v < k)

        distinct_sets = set(S.values())
        C_full = frozenset(range(k))
        candidates = list(distinct_sets) + [C_full]
        max_cliques = []
        for cs in candidates:
            is_max = True
            for cs2 in candidates:
                if cs2 != cs and cs2.issuperset(cs):
                    is_max = False
                    break
            if is_max:
                max_cliques.append(cs)

        cliques = []
        seen = set()
        for C0 in max_cliques:
            verts = set(C0)
            for s in range(k, n):
                if S[s].issuperset(C0):
                    verts.add(s)
            key = frozenset(verts)
            if key not in seen:
                seen.add(key)
                cliques.append(key)

        edge_to_cliques = {}
        for u, v in edges:
            cl = [ci for ci in range(len(cliques)) if u in cliques[ci] and v in cliques[ci]]
            edge_to_cliques[(u, v)] = cl

        unassigned = set(edge_to_cliques.keys())
        chosen = []
        while unassigned and time.time() < deadline - 5:
            best = -1
            best_count = -1
            for ci in range(len(cliques)):
                cnt = 0
                for e in unassigned:
                    if ci in edge_to_cliques[e]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best = ci
            if best < 0 or best_count <= 0:
                break
            chosen.append(best)
            for e in list(unassigned):
                if best in edge_to_cliques[e]:
                    unassigned.discard(e)

        if unassigned:
            for e in unassigned:
                for ci in edge_to_cliques[e]:
                    if ci not in chosen:
                        chosen.append(ci)
                    break

        result = []
        for ci in chosen:
            verts = set()
            for e, cl in edge_to_cliques.items():
                if ci in cl:
                    verts.add(e[0])
                    verts.add(e[1])
            if len(verts) >= 2:
                result.append(sorted(verts))
        return result

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        out.append(partition_split_graph(n, k, edges))
    return out
# EVOLVE-BLOCK-END
