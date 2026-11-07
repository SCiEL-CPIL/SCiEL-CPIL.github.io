"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        # Regenerate the exact same graph via the same RNG sequence.
        rng = Random(seed)
        # clique C = [0, k); independent set I = [k, n)
        # cross edges: for v in I, for u in C (u in 0..k-1): rng.random() < p
        cross = {}
        for v in range(k, n):
            nbr = []
            for u in range(k):
                if rng.random() < p:
                    nbr.append(u)
            cross[v] = nbr
        cliques = []
        # 1) clique on C (covers all C-C edges)
        if k >= 2:
            cliques.append(list(range(k)))
        # 2) cover cross edges: for each i in I with non-empty neighborhood,
        #    one clique {i} + N(i). Merging is always beneficial (union of
        #    cliques sharing an I-vertex is still a clique and covers a superset),
        #    so the minimum here is exactly one clique per i with N(i) non-empty.
        for i in range(k, n):
            nbr = cross[i]
            if len(nbr) >= 1:
                cliques.append([i] + list(nbr))
        # sanity: every edge must lie in exactly one clique
        # (C-C in clique0; cross edge (u,v) only in clique of v since u<C<v)
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() - _t0 > 130:
            break
        cliques = solve(n, k, p, seed)
        # type safety: plain python ints and lists
        results.append([[int(x) for x in c] for c in cliques])
    return results
# EVOLVE-BLOCK-END
