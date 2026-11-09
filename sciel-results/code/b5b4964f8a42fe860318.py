"""Clique partition via greedy max-clique packing."""
import time
import random

_DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)


def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges


def _greedy_clique_packing(adj, edges, time_budget):
    """Greedily cover edges with large cliques. Returns list of cliques."""
    n = len(adj)
    start = time.time()
    remaining = set(edges)

    # Build a fast edge -> adjacency lookup
    # For each vertex, neighbors that are still "active" (have remaining edges)
    # We track remaining edges per vertex
    rem_edges_per_vertex = [set() for _ in range(n)]
    for (u, v) in edges:
        rem_edges_per_vertex[u].add(v)
        rem_edges_per_vertex[v].add(u)

    cliques = []
    total_cliques = 0

    # Strategy: repeatedly pick a vertex with many remaining edges,
    # find a large clique in its remaining neighborhood, cover those edges.
    # Use a max-clique heuristic (Bron-Kerbosch greedy / greedy coloring bound).

    # Precompute: for efficiency, we'll use a greedy max-clique on the
    # induced subgraph of remaining edges.

    def find_large_clique(seed_vertex, max_time=0.02):
        """Find a large clique containing or near seed_vertex using greedy expansion."""
        t0 = time.time()
        # Start with seed_vertex, greedily add vertices that connect to all current
        # clique members AND have remaining edges.
        # Use greedy coloring ordering for a good clique.
        # Candidate set: neighbors of seed_vertex in remaining graph
        candidates = set(rem_edges_per_vertex[seed_vertex])
        if not candidates:
            return [seed_vertex] if rem_edges_per_vertex[seed_vertex] else []

        # Greedy: sort candidates by degree in remaining subgraph (among candidates + seed)
        # and build clique
        # Order by number of common neighbors (greedy coloring heuristic)
        def common_count(v):
            c = 0
            for w in candidates:
                if w != v and w in rem_edges_per_vertex[v]:
                    c += 1
            return c

        # Sort candidates by common_count descending (greedy)
        ordered = sorted(candidates, key=common_count, reverse=True)

        clique = [seed_vertex]
        # Greedily add vertices that connect to all in clique
        for v in ordered:
            if time.time() - t0 > max_time:
                break
            # Check v connects to all current clique members (in remaining graph)
            ok = True
            for m in clique:
                if v not in rem_edges_per_vertex[m]:
                    ok = False
                    break
            if ok:
                clique.append(v)

        return clique

    # Main loop
    while remaining:
        if time.time() > time_budget:
            break

        # Pick vertex with most remaining edges
        # To be efficient, periodically rebuild a priority
        max_deg = -1
        best_v = -1
        for v in range(n):
            d = len(rem_edges_per_vertex[v])
            if d > max_deg:
                max_deg = d
                best_v = v
            if d > 0 and max_deg >= 40:  # early stop if we have a high-degree vertex
                break

        if best_v < 0 or max_deg <= 0:
            break

        # Find a large clique containing best_v
        clique = find_large_clique(best_v)
        if len(clique) < 2:
            # Just take an edge
            # Find any remaining edge incident to best_v
            if rem_edges_per_vertex[best_v]:
                v2 = next(iter(rem_edges_per_vertex[best_v]))
                clique = [best_v, v2]
            else:
                # Find any remaining edge
                e = next(iter(remaining))
                clique = [e[0], e[1]]

        # Verify clique is valid (all edges present in remaining)
        valid = True
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                if e not in remaining:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            # Shrink clique to a valid one
            # Try removing vertices until valid
            if len(clique) >= 2:
                # Just take first two that form an edge
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if e in remaining:
                            clique = [clique[i], clique[j]]
                            break
                    if len(clique) == 2:
                        break

        if len(clique) < 2:
            # Fallback: take any remaining edge
            e = next(iter(remaining))
            clique = [e[0], e[1]]

        # Cover all edges in this clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                if e in remaining:
                    remaining.discard(e)
                    rem_edges_per_vertex[clique[i]].discard(clique[j])
                    rem_edges_per_vertex[clique[j]].discard(clique[i])

        cliques.append(clique)
        total_cliques += 1

    # Handle any remaining edges not covered (shouldn't happen, but safety)
    for e in remaining:
        cliques.append(list(e))
        total_cliques += 1

    return cliques


def construct():
    t_start = time.time()
    time_budget = _DEADLINE - 5.0  # 135s for computation

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edges = _generate_graph(n, p, seed)
        # Allocate time proportionally to graph size
        clique_list = _greedy_clique_packing(adj, edges, time_budget)
        results.append(clique_list)

    return results
