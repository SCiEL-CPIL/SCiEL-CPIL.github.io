"""Clique partition of two fixed dense random graphs."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock

    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve(n, p, seed, deadline):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]

        # candidate order: high degree first
        order = sorted(range(n), key=lambda i: (-deg[i], i))

        # remaining edges to cover
        rem = {}
        for u, v in edges:
            rem[(u, v)] = True

        cliques = []

        def try_clique(seed_v):
            """Greedy maximal clique starting from seed_v, picking neighbors that
            connect to all current members, ordered by degree."""
            clique = [seed_v]
            cand = set(adj[seed_v])
            while cand:
                # pick candidate with highest degree to all current members (all by construction)
                # just pick highest degree overall among cand
                best = max(cand, key=lambda x: deg[x])
                # ensure best connects to all in clique
                ok = all(best in adj[c] for c in clique)
                if not ok:
                    cand.discard(best)
                    continue
                clique.append(best)
                # restrict candidates to common neighbors
                new_cand = set(adj[best])
                for c in clique:
                    new_cand &= adj[c]
                cand = new_cand
            return clique

        def is_valid_clique(clique):
            if len(clique) < 2:
                return False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    if (clique[i], clique[j]) not in rem and (clique[j], clique[i]) not in rem:
                        # edge not present in original? check adj
                        if clique[j] not in adj[clique[i]]:
                            return False
            return True

        def remove_clique_edges(clique):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    key = (a, b) if a < b else (b, a)
                    if key in rem:
                        del rem[key]

        # Main loop: repeatedly find a large clique covering remaining edges
        while rem:
            if time.time() > deadline - 5:
                break
            # pick the edge with lowest "coverability" (fewest common neighbors) to force
            # coverage of hard edges first
            # choose an edge whose endpoints have few common neighbors
            # to reduce wasted large cliques
            best_edge = None
            best_score = None
            # sample a subset for speed
            keys = list(rem.keys())
            # deterministic: iterate all but cap work
            for idx, (u, v) in enumerate(keys):
                if idx % 3 == 0:  # subsample for speed
                    pass
            # pick edge with minimum |common neighbors|
            min_common = None
            for (u, v) in keys:
                cn = len(adj[u] & adj[v])
                if min_common is None or cn < min_common[0]:
                    min_common = (cn, u, v)
            if min_common is None:
                break
            _, su, sv = min_common
            # try to build a clique containing this edge
            base = [su, sv]
            cand = adj[su] & adj[sv]
            # greedy extend
            clique = base[:]
            while cand:
                best = max(cand, key=lambda x: deg[x])
                clique.append(best)
                new_cand = adj[best]
                for c in clique:
                    new_cand &= adj[c]
                cand = new_cand

            # try to maximize: also try starting from each endpoint and from high-degree
            # pick best clique (largest) among a few seeds
            candidates = [clique]
            for s in (su, sv):
                c2 = try_clique(s)
                if len(c2) >= 2:
                    candidates.append(c2)
            # choose largest
            clique = max(candidates, key=len)
            if len(clique) < 2:
                # fallback: just the edge
                clique = [su, sv]
            # ensure it covers at least the seed edge
            if su not in clique or sv not in clique:
                # rebuild to include both
                clique = [su, sv]
                cand = adj[su] & adj[sv]
                while cand:
                    best = max(cand, key=lambda x: deg[x])
                    clique.append(best)
                    new_cand = adj[best]
                    for c in clique:
                        new_cand &= adj[c]
                    cand = new_cand
            cliques.append(clique)
            remove_clique_edges(clique)

        # Cover any leftover edges individually
        leftover = list(rem.keys())
        for u, v in leftover:
            cliques.append([u, v])

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # fallback: trivial edge partition for any remaining instances
            for nn, pp, ss in ((32, 0.6, 601), (64, 0.55, 602)):
                if len(results) < 2:
                    ed = instance(nn, pp, ss)
                    results.append([list(e) for e in ed])
            break
        results.append(solve(n, p, seed, deadline))

    return results
# EVOLVE-BLOCK-END
