"""Clique partition of split graphs using optimized partitioning strategy."""
import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: 1 clique for C, 1 per cross edge
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency for I-vertices
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # For each I-vertex, find its neighbors in C
        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh

        # Strategy: For each I-vertex v, partition N_C[v] into subsets.
        # Each subset S (|S|>=1) gives clique {v} ∪ S.
        # C-C edges within S are also covered.
        # Remaining C-C edges covered by C-only cliques.

        # The key decision: how to partition each N_C[v].
        # We want to minimize: (total Type B cliques) + (C-only cliques for remaining C-C edges)

        # Since K_k can be covered by 1 clique, the baseline is:
        #   1 C-only clique + sum(|N_C[v]|) Type B cliques (each S has size 1)
        #   Total = 1 + sum(d_v)

        # Alternative: group neighbors. If v has d neighbors, putting them all in one set:
        #   1 Type B clique, consumes d(d-1)/2 C-C edges
        #   Remaining C-C edges: K_k minus those d(d-1)/2 edges
        #   Need to cover remaining with C-only cliques

        # For the complement: the remaining C-C graph has all edges EXCEPT those within any S used.
        # The min clique cover of remaining C-C graph = chromatic number of its complement.
        # The complement of remaining C-C graph = graph with edges exactly where C-C edges were consumed.

        # Let's try: for each v, use one big set (all neighbors together) if d_v >= 2.
        # This minimizes Type B cliques. Then cover remaining C-C edges.

        # But wait - if multiple v's consume overlapping C-C edges, we need to be careful.
        # Each C-C edge can only be in ONE clique total.

        # Let me think differently. The C-C edges must be partitioned among:
        #   - C-only cliques
        #   - Type B cliques (each Type B clique {v}∪S consumes C-C edges within S)

        # Constraint: no C-C edge is consumed by two different Type B cliques or by a Type B and a C-only clique.

        # This is a complex optimization. Let me try a greedy/heuristic approach:
        # 1. For each v, decide partition of N_C[v]
        # 2. Track which C-C edges are "used" by Type B cliques
        # 3. Cover remaining C-C edges with C-only cliques (minimum clique cover)

        # For minimum clique cover of a graph, we can use the fact that for the complement,
        # it's a coloring problem. For small k (up to 30), we can use a simple approach.

        # Actually, let me reconsider the structure. The C-C edges form K_k.
        # If we use Type B cliques that consume some C-C edges, the remaining C-C edges
        # form a subgraph of K_k. We need to cover this subgraph with cliques.

        # The minimum number of cliques to cover a graph G is the clique cover number θ(G).
        # For a complete graph minus some edges, this can be computed.

        # Let me try a simpler approach: 
        # Option A: 1 C-only clique (all of C) + 1 Type B clique per cross edge
        #   Total = 1 + |cross_edges|
        # Option B: For each v with d_v >= 2, use 1 Type B clique with all d_v neighbors
        #   This consumes C-C edges within N_C[v]. 
        #   Remaining C-C edges must be covered.
        #   But different v's may have overlapping neighborhoods, causing conflicts.

        # The conflict issue: if v1 and v2 both use a C-C edge (u,w) in their Type B cliques,
        # that edge would be in two cliques - INVALID.

        # So we need to ensure no C-C edge is consumed by more than one Type B clique.

        # This makes it a packing problem. Let me try:
        # For each v, the C-C edges consumed are E(N_C[v]) (all pairs within N_C[v]).
        # We need to select a subset of v's to use "big" Type B cliques such that
        # no C-C edge is in E(N_C[v1]) ∩ E(N_C[v2]) for selected v1, v2.

        # This is getting complex. Let me try a different approach:
        # Use the fact that for the C-C edges, the optimal cover is 1 clique (the whole C).
        # The question is whether we can save Type B cliques by "absorbing" some C-C edges.

        # Each Type B clique {v}∪S saves (|S|-1) Type B cliques compared to using |S| separate
        # {v}∪{u} cliques, but it consumes C-C edges within S.

        # The C-C edges consumed must not conflict with other Type B cliques or the C-only clique.

        # Let me try: for each v, use one Type B clique with all neighbors (if d_v >= 2).
        # Check for conflicts. If conflicts exist, fall back to individual edges.

        # Actually, the simplest improvement: if d_v = 1, we must use {v, u} (1 clique, 1 cross edge).
        # If d_v >= 2, we can use {v} ∪ N_C[v] (1 clique, d_v cross edges, d_v(d_v-1)/2 C-C edges).
        # But this C-C edge consumption might conflict.

        # Let me just try the "all in one" strategy and check validity.
        # If it's invalid, fall back to the safe strategy.

        # Safe strategy: 1 C-only clique + 1 per cross edge
        safe_cliques = [list(range(k))]
        cross_edges = [(u, v) for (u, v) in edge_set if u >= k or v >= k]
        for (u, v) in cross_edges:
            safe_cliques.append([u, v])
        safe_count = len(safe_cliques)

        # Try improved strategy
        best_cliques = safe_cliques
        best_count = safe_count

        # For each v, if d_v >= 2, consider using one big clique
        # Track C-C edges used by Type B cliques
        used_cc_edges = set()
        improved_cliques = []
        valid = True

        # Sort I-vertices by degree (descending) to pack more
        I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)

        for v in I_verts:
            neigh = list(N_C[v])
            d = len(neigh)
            if d == 0:
                continue
            elif d == 1:
                improved_cliques.append([v, neigh[0]])
            else:
                # Check if C-C edges within neigh are all unused
                cc_in_neigh = set()
                for i in range(d):
                    for j in range(i + 1, d):
                        a, b = neigh[i], neigh[j]
                        e = (a, b) if a < b else (b, a)
                        cc_in_neigh.add(e)

                if cc_in_neigh & used_cc_edges:
                    # Conflict - use individual edges
                    for u in neigh:
                        improved_cliques.append([v, u])
                else:
                    # Use big clique
                    improved_cliques.append([v] + neigh)
                    used_cc_edges |= cc_in_neigh

        # Now cover remaining C-C edges with C-only cliques
        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_cc = all_cc_edges - used_cc_edges

        if remaining_cc:
            # Need to cover remaining_cc with C-only cliques
            # Minimum clique cover of the graph with edge set remaining_cc
            # For a subgraph of K_k, the min clique cover can be found by
            # coloring the complement.
            # Complement of remaining_cc graph: edges are all_cc_edges - remaining_cc = used_cc_edges
            # Wait, no. The graph G has vertices C and edges remaining_cc.
            # We want min cliques covering all edges of G.
            # This is the edge clique cover number.
            # 
            # For a complete graph, it's 1. For a general graph, it's NP-hard.
            # But for our case, remaining_cc is a subgraph of K_k where some edges are missing
            # (those consumed by Type B cliques).
            #
            # Simple approach: use a greedy coloring of the complement to get clique cover.
            # Or just use a simple heuristic.
            #
            # Actually, for the remaining C-C edges, we can use a simple strategy:
            # Find the maximum clique in the remaining graph, remove its edges, repeat.
            # Or use a simple 1-clique if the remaining graph is complete.
            #
            # Let me check if remaining_cc is complete (i.e., no C-C edges were consumed).
            if len(remaining_cc) == len(all_cc_edges):
                # No C-C edges consumed, use 1 clique
                improved_cliques.append(list(range(k)))
            else:
                # Need to cover remaining_cc. Use a greedy approach.
                # Build adjacency for remaining_cc
                rem_adj = {u: set() for u in range(k)}
                for (a, b) in remaining_cc:
                    rem_adj[a].add(b)
                    rem_adj[b].add(a)
                
                # Greedy: repeatedly find a large clique in remaining graph
                covered = set()
                while remaining_cc - covered:
                    # Find a clique that covers many uncovered edges
                    # Simple: take a vertex with high degree in remaining graph, 
                    # extend to a clique
                    best_clique = None
                    best_edges = -1
                    for start in range(k):
                        if not rem_adj[start]:
                            continue
                        # Try to build a clique starting from start
                        clique = [start]
                        for w in range(start + 1, k):
                            if all(w in rem_adj[x] for x in clique):
                                clique.append(w)
                        # Count uncovered edges in this clique
                        cl_edges = set()
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                a, b = clique[i], clique[j]
                                e = (a, b) if a < b else (b, a)
                                cl_edges.add(e)
                        uncov = cl_edges - covered
                        if len(uncov) > best_edges:
                            best_edges = len(uncov)
                            best_clique = clique
                    
                    if best_clique is None or best_edges <= 0:
                        # Shouldn't happen, but fallback
                        for (a, b) in remaining_cc - covered:
                            improved_cliques.append([a, b])
                            covered.add((a, b))
                        break
                    
                    improved_cliques.append(best_clique)
                    for i in range(len(best_clique)):
                        for j in range(i + 1, len(best_clique)):
                            a, b = best_clique[i], best_clique[j]
                            e = (a, b) if a < b else (b, a)
                            covered.add(e)

        improved_count = len(improved_cliques)

        if improved_count < best_count:
            best_cliques = improved_cliques
            best_count = improved_count

        results.append(best_cliques)

    return results
