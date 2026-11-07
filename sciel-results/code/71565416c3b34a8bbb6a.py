"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency for the clique part (first k vertices)
        # and for each independent vertex, its neighbors in the clique
        cross_neighbors = [set() for _ in range(k, n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_neighbors[v - k].add(u)

        cliques = []

        # Step 1: For each independent vertex, form clique {v} ∪ N(v)
        # This covers all cross edges exactly once
        for i in range(k, n):
            idx = i - k
            neighbors = cross_neighbors[idx]
            if len(neighbors) >= 1:  # clique must have at least 2 vertices
                clique = [i] + sorted(neighbors)
                cliques.append(clique)

        # Step 2: Cover edges within the clique part (first k vertices)
        # Build adjacency for the subgraph on vertices 0..k-1
        adj = [set() for _ in range(k)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)

        # Greedy: repeatedly pick vertex with max degree, form clique with it and all neighbors
        remaining_adj = [set(s) for s in adj]
        while True:
            # Find vertex with max remaining degree
            max_deg = -1
            max_v = -1
            for v in range(k):
                d = len(remaining_adj[v])
                if d > max_deg:
                    max_deg = d
                    max_v = v
            if max_deg <= 0:
                break

            # Form clique: max_v and all its remaining neighbors
            neighbors = list(remaining_adj[max_v])
            clique = [max_v] + sorted(neighbors)
            if len(clique) >= 2:
                cliques.append(clique)

            # Remove all edges in this clique from remaining_adj
            clique_set = set(clique)
            for u in clique:
                for v in list(remaining_adj[u]):
                    if v in clique_set:
                        remaining_adj[u].discard(v)
                        remaining_adj[v].discard(u)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        # Ensure all vertices are Python int and cliques are lists
        clean_cliques = [[int(v) for v in clique] for clique in cliques]
        results.append(clean_cliques)

    return results
# EVOLVE-BLOCK-END
