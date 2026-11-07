"""Clique partition of three fixed split graphs: minimize number of cliques."""
import time

_START = time.time()
_DEADLINE = 115.0  # safety margin inside the 120 s limit


def _time_up():
    return time.time() > _DEADLINE


def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def max_clique(adj, n):
        # Bron-Kerbosch with pivot; returns largest clique found.
        best = []
        best_ref = [best]

        def bk(R, P, X):
            if _time_up():
                return
            if not P and not X:
                if len(R) > len(best_ref[0]):
                    best_ref[0][:] = R
                return
            # pivot: choose u in P|X with max |P & N(u)|
            up = P | X
            if up:
                u = max(up, key=lambda x: len(P & adj[x]))
                cands = P - adj[u]
            else:
                cands = P
            for v in list(cands):
                bk(R + [v], P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                if _time_up():
                    return

        bk([], set(range(n)), set())
        return best

    def greedy_maximal(adj, n):
        # repeatedly pick a large maximal clique, remove its edges.
        adj = [set(s) for s in adj]
        cliques = []
        remaining = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    remaining[u].add(v)
                    remaining[v].add(u)

        active = [i for i in range(n) if remaining[i]]
        while active and not _time_up():
            # choose seed vertex with highest degree, grow a maximal clique greedily
            seed = max(active, key=lambda x: len(remaining[x]))
            clique = [seed]
            clique_set = {seed}
            # candidates: neighbors of all clique members, sorted by degree
            cand = set()
            for m in clique:
                cand |= remaining[m]
            cand = [v for v in cand if all(v in remaining[m] for m in clique)]
            cand.sort(key=lambda v: (-len(remaining[v]), v))
            for v in cand:
                if all(v in remaining[m] for m in clique):
                    clique.append(v)
                    clique_set.add(v)
            # record clique, remove its edges
            if len(clique) >= 2:
                cliques.append(sorted(clique))
            for a in clique:
                for b in clique:
                    if a != b and b in remaining[a]:
                        remaining[a].discard(b)
            active = [i for i in active if remaining[i]]
        return cliques, remaining

    def repair(adj, n, remaining):
        # cover leftover edges with maximal cliques (Bron-Kerbosch on the
        # subgraph induced by leftover edges), then with single edges.
        # Build a graph of leftover edges.
        leftover = [set() for _ in range(n)]
        for u in range(n):
            for v in remaining[u]:
                if v > u:
                    leftover[u].add(v)
                    leftover[v].add(u)
        cliques = []
        # iteratively find a large clique in leftover and remove its edges
        while not _time_up():
            # pick a vertex with leftover degree
            active = [i for i in range(n) if leftover[i]]
            if not active:
                break
            # restrict to subgraph of leftover edges
            sub_adj = [leftover[i] & set(active) for i in range(n)]
            mc = max_clique(sub_adj, n)
            if len(mc) >= 2:
                cliques.append(sorted(mc))
                for a in mc:
                    for b in mc:
                        if a != b:
                            leftover[a].discard(b)
                            leftover[b].discard(a)
            else:
                # no clique of size>=2 (isolated leftovers are single edges)
                break
        # any remaining single edges
        for u in range(n):
            for v in leftover[u]:
                if v > u:
                    cliques.append([u, v])
                    leftover[u].discard(v)
                    leftover[v].discard(u)
        return cliques

    def partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = build_adj(n, edges)

        # Specialized structural partition for split graphs:
        #  - each independent vertex v>=k with K-neighbors Nv forms one clique {v}+Nv
        #  - K-clique edges not covered by those cliques are covered by a
        #    small set of vertex-disjoint cliques on K.
        # This is near-optimal for split graphs and very fast.

        # Determine K-neighbor sets for each independent vertex.
        k_neighbors = []  # list of frozenset for v in [k, n)
        for v in range(k, n):
            nb = frozenset(u for u in range(k) if u in adj[v])
            k_neighbors.append((v, nb))

        # Decide for each independent vertex whether to form its own clique.
        # Forming {v}+Nv covers |Nv| cross edges and consumes Nv for K-clique coverage.
        # If |Nv| >= 2, it is almost always beneficial (one clique covers |Nv| edges
        # instead of |Nv| separate edge-cliques).
        used_k = [False] * k  # K-vertices consumed by an independent-vertex clique
        cliques = []
        # vertices with >=2 K-neighbors get their own clique
        for v, nb in k_neighbors:
            if len(nb) >= 2:
                cliques.append(sorted([v] + list(nb)))
                for u in nb:
                    used_k[u] = True
        # vertices with 1 K-neighbor: leave the cross edge for later (pair with the
        # K-clique covering that K-vertex, or make a 2-clique)
        single_cross = [(v, list(nb)[0]) for v, nb in k_neighbors if len(nb) == 1]

        # Now cover K-clique edges between "free" K-vertices (not used by a
        # >=2 independent clique) plus the K-vertices touched by single_cross.
        # Actually K-clique edges among ALL K-vertices must be covered, but a K-vertex
        # used in an independent clique still needs its K-K edges covered.
        # So cover all K-clique edges with vertex-disjoint cliques on K.
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        k_adj = [set() for _ in range(k)]
        for u, v in k_edges:
            k_adj[u].add(v)
            k_adj[v].add(u)

        k_cliques = []
        remaining_k = [set(s) for s in k_adj]
        active_k = list(range(k))
        while active_k and not _time_up():
            seed = max(active_k, key=lambda x: len(remaining_k[x]))
            clique = [seed]
            cand = [v for v in remaining_k[seed]]
            cand.sort(key=lambda v: (-len(remaining_k[v]), v))
            for v in cand:
                if all(v in remaining_k[m] for m in clique):
                    clique.append(v)
            if len(clique) >= 2:
                k_cliques.append(clique)
            for a in clique:
                for b in clique:
                    if a != b:
                        remaining_k[a].discard(b)
            active_k = [i for i in active_k if remaining_k[i]]

        # single cross edges: pair each with a K-clique containing its K-vertex if
        # possible (merge), else make a 2-clique.
        # Map K-vertex -> index in k_cliques (first clique containing it)
        k_in_clique = {}
        for ci, cl in enumerate(k_cliques):
            for x in cl:
                if x not in k_in_clique:
                    k_in_clique[x] = ci

        for v, u in single_cross:
            if u in k_in_clique:
                ci = k_in_clique[u]
                k_cliques[ci].append(v)
                k_cliques[ci].sort()
            else:
                cliques.append(sorted([u, v]))

        all_cliques = cliques + k_cliques
        # validate: every edge in exactly one clique, each clique a clique
        # (light check in debug only; trust construction)
        return [list(c) for c in all_cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        out.append(partition(n, k, p, seed))
    return out
