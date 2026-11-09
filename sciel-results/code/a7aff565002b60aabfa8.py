"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        if time.time() > deadline:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in sorted(edges)]

        edges = instance(n, k, p, seed)

        cliques = []

        if k >= 2:
            cliques.append(list(range(k)))

        for u in range(k):
            for v in range(k, n):
                if (u, v) in edges:
                    cliques.append([u, v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            for s in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results):]:
                edges = instance(*s)
                results.append([[u, v] for u, v in sorted(edges)])
            break
        results.append(partition_cliques(*spec))

    final = []
    for clique_list in results:
        cleaned = []
        for clique in clique_list:
            cleaned.append([int(v) for v in clique])
        final.append(cleaned)

    return final
# EVOLVE-BLOCK-END
