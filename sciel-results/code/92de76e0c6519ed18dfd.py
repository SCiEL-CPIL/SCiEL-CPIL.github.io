"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def construct():
    start_time = time.time()
    deadline = 125.0  # explicit guard, safety margin before the 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    # ---- Maximum clique via Bron-Kerbosch with pivot (deterministic) ----
    def max_clique(adj, vertices):
        best = [0]
        best_size = 0
        vertices = list(vertices)

        def bk(R, P, X):
            nonlocal best, best_size
            if not P and not X:
                if len(R) > best_size:
                    best_size = len(R)
                    best = list(R)
                return
            # pivot: pick u in P∪X maximizing |N(u) ∩ P|
            union = P | X
            if union:
                pivot = max(union, key=lambda u: len(adj[u] & P))
                candidates = P - adj[pivot]
            else:
                candidates = set(P)
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(set(), set(vertices), set())
        return best

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # adjacency sets
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set(edges)
        cliques = []

        def remove_clique_edges(clique):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    remaining.discard((a, b) if a < b else (b, a))

        # ---- Step 1: cover cross edges, one clique per independent vertex ----
        # For each independent vertex w, its neighbors in the clique part form a
        # clique; adding w gives a clique that covers all (u,w) edges AND all
        # (u,v) edges among those neighbors.  Pick the largest valid neighbor
        # subset (by default all neighbors) to cover the most internal C-edges.
        indep = list(range(k, n))
        # process higher-degree first to cover more internal edges early
        indep.sort(key=lambda w: len(adj[w] & set(range(k))), reverse=True)

        for w in indep:
            if time.time() > deadline:
                break
            neighbors = [u for u in range(k) if (u, w) in remaining]
            if not neighbors:
                continue
            # neighbors is a clique (subset of the K_k clique part).
            # Largest valid subset = all of them.
            clique = neighbors + [w]
            cliques.append(clique)
            remove_clique_edges(clique)

        # ---- Step 2: cover remaining edges within the clique part (0..k-1) ----
        # The residual graph G' on 0..k-1 is a subgraph of K_k.
        # Repeatedly find a maximum clique in G' and remove its edges.
        while remaining:
            if time.time() > deadline:
                break
            # restrict to vertices that still have a remaining edge
            active = set()
            for u, v in remaining:
                active.add(u)
                active.add(v)
            # build adjacency of the residual graph restricted to active
            radj = {v: set() for v in active}
            for u, v in remaining:
                radj[u].add(v)
                radj[v].add(u)
            clique = max_clique(radj, active)
            if len(clique) < 2:
                # should not happen if remaining is nonempty, but be safe
                e = next(iter(remaining))
                clique = [e[0], e[1]]
            cliques.append(clique)
            remove_clique_edges(clique)

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            # fallback: individual edges (valid, each size 2)
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
        else:
            res = solve_split(*spec)
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)

    return results
# EVOLVE-BLOCK-END
