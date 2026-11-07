def construct():
    import time
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve(n, k, p, seed, deadline):
        rng = Random(seed)

        # Build cross adjacency
        adj_cross = []
        for v in range(k, n):
            neighbors = set()
            for u in range(k):
                if rng.random() < p:
                    neighbors.add(u)
            adj_cross.append(neighbors)

        # Core is a complete graph on vertices 0..k-1
        # Each independent vertex i (k <= i < n) connects to some subset of core
        # A valid clique can contain:
        #   - Multiple core vertices (must be pairwise adjacent = always true since core is complete)
        #   - At most 1 independent vertex (since independent vertices aren't connected to each other)
        # So each clique is either:
        #   - A set of >=2 core vertices (pure core clique)
        #   - A set of >=1 core vertices + exactly 1 independent vertex connected to all of them
        #     (but needs at least 2 total vertices, so if 1 core + 1 ind, that's 2)
        #
        # Strategy: 
        # 1. For each independent vertex, assign it to a clique with some core vertices
        # 2. The core vertices in that clique must all be neighbors of the independent vertex
        # 3. Each edge must appear in exactly one clique
        # 4. Core-core edges: each must be in exactly one pure core clique
        # 5. Cross edges: edge (u,v) where u in core, v in ind: must be in exactly one clique
        #    that contains both u and v
        #
        # Key insight: if we put independent vertex v in a clique with core set S, then:
        #   - All edges (u,v) for u in S are covered by this clique
        #   - All edges (u,w) for u,w in S are ALSO covered by this clique
        #   - This means those core-core edges can't be in another clique
        #
        # So the problem is: partition the core-core edges among:
        #   - Pure core cliques
        #   - Cliques that include an independent vertex
        # And assign each cross edge to the clique containing its independent endpoint.
        #
        # For each independent vertex v with neighbor set N_v:
        #   If v is in a clique with core set S_v (subset of N_v, |S_v| >= 1 if we want cross edges,
        #   or we could have v alone but that's only 1 vertex - invalid)
        #   Actually v must be in a clique with at least 1 other vertex. If S_v is non-empty, 
        #   the clique is S_v ∪ {v} and has |S_v|+1 >= 2 vertices.
        #   The cross edges covered are (u,v) for u in S_v.
        #   The core-core edges covered are (u,w) for u,w in S_v, u<w.
        #
        # So we need to:
        # 1. For each independent vertex v, choose S_v ⊆ N_v, |S_v| >= 1
        # 2. The sets S_v and the pure core cliques together must partition all core-core edges
        #    (each core-core edge appears in exactly one S_v or one pure core clique)
        # 3. All cross edges are covered: for each (u,v), u in core, v in ind, edge is in S_v∪{v}
        #    iff u ∈ S_v. So we need u ∈ S_v for ALL neighbors u of v. 
        #    Wait no - every cross edge must be in EXACTLY one clique. The only clique that can
        #    contain edge (u,v) is one containing both u and v. The only vertex v appears in is 
        #    its assigned clique S_v∪{v}. So edge (u,v) is covered iff u ∈ S_v.
        #    Therefore we MUST have S_v = N_v for all v! Because every neighbor u of v must have
        #    edge (u,v) covered, which requires u ∈ S_v.
        #
        # So S_v = N_v is forced! Each independent vertex v must be in a clique with ALL its core neighbors.
        #
        # Then the core-core edges covered by v's clique are all edges within N_v.
        # The remaining core-core edges (not in any N_v) must be covered by pure core cliques.
        #
        # But wait - can two independent vertices v1, v2 both be in cliques that share core vertices?
        # Yes! The core-core edge (u,w) would be in BOTH cliques, violating "exactly one".
        #
        # So we need: for any pair of independent vertices v1, v2, N_v1 ∩ N_v2 must have
        # no edges... but core is complete, so any two vertices in the intersection form an edge.
        # This means N_v1 and N_v2 must be DISJOINT for all pairs v1 ≠ v2!
        #
        # But that's generally not the case. So the constraint is:
        # The sets {E(N_v) : v in independent} must be pairwise edge-disjoint, where
        # E(S) is the set of core-core edges within S.
        # Two cliques S_v ∪ {v} and S_w ∪ {w} share a core-core edge iff |N_v ∩ N_w| >= 2.
        #
        # So if two independent vertices share 2+ core neighbors, their cliques overlap on core edges.
        # This is a problem!
        #
        # Resolution: we can't put v in a clique with ALL of N_v if that causes overlap.
        # But we MUST cover all cross edges. Edge (u,v) must be in a clique containing both u and v.
        # The only clique containing v is v's assigned clique. So u must be in S_v.
        #
        # Unless... v is in multiple cliques? No, the partition assigns each vertex to cliques,
        # but actually re-reading: "every edge must lie in exactly one listed clique". 
        # Vertices can appear in multiple cliques! It's edges that are partitioned.
        #
        # Oh wait, let me re-read: "Every listed set must be a clique with at least two vertices 
        # and every edge must lie in exactly one listed clique."
        #
        # So vertices CAN appear in multiple cliques. Each edge must be in exactly one clique.
        # This changes everything!
        #
        # So for independent vertex v, it can appear in multiple cliques. 
        # Edge (u,v) is in exactly one clique, which must contain both u and v.
        # 
        # So we can split v's neighbors among different cliques!
        # E.g., if v has neighbors {a, b, c}, we could put v in clique {a, b, v} and clique {c, v}.
        # Then edge (a,v) is in clique 1, edge (b,v) is in clique 1, edge (c,v) is in clique 2.
        # Edge (a,b) is in clique 1. Edge (a,c) and (b,c) are not in any clique with v,
        # so they must be in pure core cliques.
        #
        # This is much more flexible! Now the problem is:
        # For each independent vertex v with neighbor set N_v, partition N_v into subsets
        # S_v1, S_v2, ..., S_vm (each non-empty). For each Sj, form clique Sj ∪ {v}.
        # This covers cross edges (u,v) for u in Sj, and core-core edges within Sj.
        # The remaining core-core edges (not covered by any such clique) must be covered
        # by pure core cliques.
        # Goal: minimize total number of cliques.
        #
        # Each independent vertex v with |N_v| = d needs at least 1 clique (if we put all of N_v
        # in one subset) or up to d cliques (one per neighbor, but each needs >=1 other vertex...
        # a single neighbor gives clique {u, v} which is 2 vertices, valid).
        #
        # To minimize cliques, we want to maximize the size of each Sj, which means using
        # as few subsets per vertex as possible. Using 1 subset per vertex (S_v = N_v) uses
        # the fewest cliques for cross edges, but may cause many core-core edges to be "used up"
        # in these cliques, leaving the rest to pure core cliques.
        #
        # Actually, using more subsets per vertex doesn't directly increase cross-edge cliques
        # unless we need it to avoid overlap. Since vertices can be in multiple cliques, there's
        # no overlap issue! Edge (u,w) in core is in exactly one clique: either in some Sj∪{v} 
        # where u,w ∈ Sj, or in a pure core clique.
        #
        # So the optimal is: for each independent vertex v, use ONE clique S_v ∪ {v} where S_v = N_v.
        # This uses 1 clique per independent vertex (with neighbors) for all cross edges.
        # Then cover remaining core-core edges with pure core cliques.
        #
        # Total cliques = (number of independent vertices with at least 1 neighbor) + 
        #                  (minimum clique cover of remaining core edges)
        #
        # This is what the original code was trying to do! Let me implement it correctly.

        # For each independent vertex, create a clique with all its core neighbors
        # Track which core-core edges are used
        used_core_edges = set()
        cliques = []

        for idx, neighbors in enumerate(adj_cross):
            if not neighbors:
                continue
            v = k + idx
            S = sorted(neighbors)
            cliques.append(S + [v])
            # Mark core-core edges within S as used
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    used_core_edges.add((S[i], S[j]))

        # Cover remaining core-core edges with pure core cliques
        all_core_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_core_edges.add((u, w))

        remaining_edges = all_core_edges - used_core_edges

        # Greedy clique cover of remaining edges
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover each edge as a 2-clique
                for e in remaining_edges:
                    cliques.append(list(e))
                break

            # Build adjacency on remaining edges
            adj_rem = {v: set() for v in range(k)}
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)

            # Find a maximal clique
            S = set()
            for v in range(k):
                if not adj_rem[v]:
                    continue
                ok = True
                for u in S:
                    if v not in adj_rem[u]:
                        ok = False
                        break
                if ok:
                    S.add(v)

            if len(S) < 2:
                # Take any edge
                u, v = min(remaining_edges)
                S = {u, v}

            cliques.append(sorted(S))
            for u in S:
                for w in S:
                    if u < w:
                        remaining_edges.discard((u, w))

        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve(*spec, deadline))

    final_results = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_cliques.append([int(v) for v in clique])
        final_results.append(clean_cliques)

    return final_results
