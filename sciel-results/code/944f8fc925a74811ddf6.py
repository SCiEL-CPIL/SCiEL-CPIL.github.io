"""Rewritten program: clique partition via greedy sequential coloring heuristic."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

    def instance_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(n, p, seed):
        edges = instance_edges(n, p, seed)
        if not edges:
            return []

        # Build adjacency for the original graph
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute for each vertex, the set of other vertices it's connected to
        # For clique checking: a set S is a clique iff every pair in S is an edge.
        # We'll maintain for each clique (as a set of vertices) whether it's still a clique.

        # Strategy: assign each edge to the smallest-indexed clique whose vertex set
        # remains a clique after adding both endpoints.
        # A clique is represented by its vertex set. When we add edge (u,v) to clique C,
        # the new vertex set is C ∪ {u, v}. This must be a clique.
        # That means u must be adjacent to every vertex in C, and v must be adjacent to every vertex in C,
        # and u must be adjacent to v (which it is, since (u,v) is an edge).

        # Greedy: process edges in some order, assign each to the first compatible clique.
        # To minimize number of cliques, we want to pack edges into existing cliques as much as possible.

        # Order edges by some heuristic: perhaps by degree of endpoints, or just natural order.
        # Let's try processing edges in the order they appear.

        cliques = []  # list of sets of vertices
        edge_to_clique = [None] * len(edges)

        for idx, (u, v) in enumerate(edges):
            placed = False
            for ci, clique in enumerate(cliques):
                # Check if adding u and v to clique keeps it a clique
                # u must be adjacent to all in clique, v must be adjacent to all in clique
                ok = True
                for w in clique:
                    if w != u and w != v:
                        if v not in adj.get(w, set()) or u not in adj.get(w, set()):
                            ok = False
                            break
                if ok:
                    # Also need u adjacent to v (guaranteed) and u,v adjacent to each other
                    # And u adjacent to all in clique, v adjacent to all in clique
                    # We checked that. Also need that within clique, all pairs are edges (invariant)
                    # So new set is clique.
                    cliques[ci] = clique | {u, v}
                    edge_to_clique[idx] = ci
                    placed = True
                    break
            if not placed:
                cliques.append({u, v})
                edge_to_clique[idx] = len(cliques) - 1

        # Convert cliques to lists
        return [list(c) for c in cliques]

    return [solve(32, 0.6, 601), solve(64, 0.55, 602)]
# EVOLVE-BLOCK-END
