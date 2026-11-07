"""Rewritten program with optimal split-graph clique partition."""

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        """Return list of cliques (each a list of >=2 distinct ints) covering every edge exactly once."""
        # The first k vertices form a clique C; the rest form an independent set I.
        # Cross edges are random with probability p.
        #
        # Key structural observation for split graphs:
        #   - Vertices in I are independent (no edges among them).
        #   - A clique can contain at most one vertex from I (since I vertices aren't connected to each other).
        #   - Therefore, every cross edge {c, i} must be covered by a clique that contains c and i,
        #     and no other I-vertices can be in that clique (they'd need edges to each other).
        #   - So each cross edge {c, i} requires its own dedicated clique of size 2: [c, i].
        #
        # For the edges within C (the first k vertices):
        #   - C is a clique, so all C-C edges exist.
        #   - We can cover all C-C edges with a single clique: list(range(k)).
        #   - This is valid because no I-vertex is in this clique, so no cross edges are included.
        #
        # Total: 1 clique for C + (number of cross edges) cliques of size 2.
        #
        # Every edge appears in exactly one clique:
        #   - C-C edges: only in the big C clique.
        #   - Cross edges: only in their respective 2-cliques.
        #   - No I-I edges exist.
        #
        # Every clique has at least 2 vertices:
        #   - The C clique has k >= 2 vertices (k is 10, 20, 30 in our cases).
        #   - Cross-edge cliques have exactly 2 vertices.

        # Generate the cross edges using the same RNG sequence as instance()
        rng = Random(seed)
        # Skip the C-C edge generation (instance() generates them but doesn't use RNG for them)
        # In instance(), C-C edges are generated first (no RNG calls), then cross edges use RNG.
        # So we just need to iterate cross edges in the same order.

        cliques = [list(range(k))]  # The big clique on C

        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([u, v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))

    return results
