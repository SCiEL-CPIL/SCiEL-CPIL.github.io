"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_clique_partition(n, edges, deadline):
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        total_edges = len(edge_set)
        if total_edges == 0:
            return []

        # Find all maximal cliques using Bron-Kerbosch
        max_cliques = []

        def bron_kerbosch(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_set = P | X
            if pivot_set:
                pivot = max(pivot_set, key=lambda v: len(adj[v] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bron_kerbosch(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        bron_kerbosch(set(), set(range(n)), set())

        # Filter to cliques with at least 2 vertices
        max_cliques = [c for c in max_cliques if len(c) >= 2]

        if not max_cliques:
            # Fallback: each edge is its own clique
            return [[u, v] for u, v in edges]

        # Greedy set cover: pick clique covering most uncovered edges
        # Represent edges as a set for fast membership
        remaining_edges = set(edge_set)
        selected = []

        # Precompute edge lists for each clique
        clique_edges = []
        for c in max_cliques:
            verts = list(c)
            es = frozenset((min(u, v), max(u, v)) for u, v in itertools.combinations(verts, 2))
            clique_edges.append(es)

        # Greedy: repeatedly pick clique that covers most uncovered edges
        while remaining_edges:
            best_idx = -1
            best_count = 0
            for i, es in enumerate(clique_edges):
                count = len(es & remaining_edges)
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(max_cliques[best_idx])
            remaining_edges -= clique_edges[best_idx]

        # If greedy didn't cover everything (shouldn't happen if all edges are in some max clique),
        # add remaining edges as 2-vertex cliques
        if remaining_edges:
            for u, v in remaining_edges:
                selected.append(frozenset([u, v]))

        # Convert to lists of ints
        result = [sorted(list(c)) for c in selected]
        return result

    # Main logic
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result (each edge as a clique)
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        partition = find_clique_partition(n, edges, deadline)
        results.append(partition)

    return results
# EVOLVE-BLOCK-END
