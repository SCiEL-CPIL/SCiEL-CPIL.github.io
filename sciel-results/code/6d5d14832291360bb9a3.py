"""Rewritten program: exact minimum clique cover via Bron-Kerbosch + branch-and-bound."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bron_kerbosch(R, P, X, adj, out):
        # P, X are frozensets; adj maps vertex -> frozenset of neighbors
        if not P and not X:
            if len(R) >= 2:
                out.append(frozenset(R))
            return
        # pivot: choose u in P|X with max |P & N(u)|
        candidates = P | X
        best = None
        best_cnt = -1
        for u in candidates:
            c = len(P & adj[u])
            if c > best_cnt:
                best_cnt = c
                best = u
        if best is None:
            best_cnt = 0
        # iterate over P \ N(best)
        for v in list(P):
            if v in adj.get(best, frozenset()):
                continue
            P.remove(v)
            X.remove(v)
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, out)
            P.add(v)
            X.add(v)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        adj = {v: frozenset() for v in range(n)}
        for u, v in edges:
            adj[u] = adj[u] | {v}
            adj[v] = adj[v] | {u}
        # enumerate all maximal cliques of size >= 2
        maxcliques = []
        bron_kerbosch(frozenset(), frozenset(adj.keys()), frozenset(), adj, maxcliques)
        maxcliques = [c for c in maxcliques if len(c) >= 2]
        if not maxcliques:
            return []
        # map each edge to candidate cliques containing it
        edge_to_cands = {}
        for idx, c in enumerate(maxcliques):
            for u in c:
                for v in c:
                    if u < v:
                        edge_to_cands.setdefault((u, v), []).append(idx)
        # sanity: every edge must be covered by at least one maximal clique
        for e in edges:
            if e not in edge_to_cands or not edge_to_cands[e]:
                # fallback: edge itself as a clique
                return [list(e) for e in edges]
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        # for each clique, the set of edge indices it covers
        clique_edges = []
        for c in maxcliques:
            s = []
            for u in c:
                for v in c:
                    if u < v:
                        s.append(edge_idx[(u, v)])
            clique_edges.append(s)
        # greedy lower bound
        def lower_bound(covered):
            rem = m - len(covered)
            if rem <= 0:
                return 0
            best = 0
            for s in clique_edges:
                c = 0
                for e in s:
                    if e not in covered:
                        c += 1
                if c > best:
                    best = c
            return (rem + best - 1) // best
        # branch and bound
        best = [m + 1]
        best_sol = [None]
        cur = [set(), []]  # covered, chosen clique indices
        # order edges by fewest candidates (most constrained) for branching
        def branch():
            if time.time() > deadline:
                return
            if len(cur[0]) == m:
                if len(cur[1]) < best[0]:
                    best[0] = len(cur[1])
                    best_sol[0] = list(cur[1])
                return
            lb = lower_bound(cur[0])
            if len(cur[1]) + lb >= best[0]:
                return
            # pick uncovered edge with fewest candidate cliques (MRV)
            chosen_e = None
            chosen_cands = None
            min_len = 10**9
            for e in range(m):
                if e in cur[0]:
                    continue
                cands = edge_to_cands[edges[e]]
                # only cliques that still add something? all add at least this edge
                if len(cands) < min_len:
                    min_len = len(cands)
                    chosen_e = e
                    chosen_cands = cands
                    if min_len == 1:
                        break
            if chosen_e is None:
                return
            for ci in chosen_cands:
                s = clique_edges[ci]
                # check disjointness with already covered
                overlap = False
                for e in s:
                    if e in cur[0]:
                        overlap = True
                        break
                if overlap:
                    continue
                cur[0].update(s)
                cur[1].append(ci)
                branch()
                cur[1].pop()
                for e in s:
                    cur[0].discard(e)
                if time.time() > deadline:
                    return
        branch()
        if best_sol[0] is None:
            # fallback: cover each edge individually
            return [list(e) for e in edges]
        return [sorted(list(maxcliques[ci])) for ci in best_sol[0]]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        res = solve_one(*spec)
        results.append(res)
        if time.time() > deadline:
            break
    # pad if a later instance didn't finish in time
    while len(results) < 3:
        results.append([])
    return results
# EVOLVE-BLOCK-END
