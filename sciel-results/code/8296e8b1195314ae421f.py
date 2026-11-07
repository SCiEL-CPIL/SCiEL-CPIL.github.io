"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def get_cliques_for_instance(n, k, p, seed):
        rng = Random(seed)
        # Neighbors of each independent-set vertex among the k-clique vertices.
        neighbors = []
        for v in range(k, n):
            nbrs = [u for u in range(k) if rng.random() < p]
            neighbors.append((v, nbrs))

        cliques = []
        # One clique for the full k-clique: covers every internal edge exactly once.
        cliques.append(list(range(k)))
        # One clique per independent vertex that has at least one cross edge:
        # {v} U N(v) covers all of v's cross edges, and no other cross edge
        # uses v, so each cross edge lies in exactly one listed clique.
        for v, nbrs in neighbors:
            if nbrs:
                cliques.append([v] + nbrs)
        # Every listed set is a clique (>=2 vertices); every edge in exactly one.
        return cliques

    return [get_cliques_for_instance(*spec)
            for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))]
# EVOLVE-BLOCK-END
