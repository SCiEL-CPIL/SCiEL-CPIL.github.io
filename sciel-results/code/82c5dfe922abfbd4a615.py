def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
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

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for a, b in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # C = {0, ..., k-1}, I = {k, ..., n-1}
        # C-C edges
        cc_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                cc_edges.add((u, w))

        # Cross edges: (u, v) with u < k <= v
        cross_edges = set()
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges.add((a, b))
            elif b < k and a >= k:
                cross_edges.add((b, a))

        # For each I-vertex v, its neighbor set in C
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in cross_edges)

        # Baseline: 1 C-clique + per-cross-edge 2-cliques
        base_cliques = [list(range(k))]
        for (u, v) in sorted(cross_edges):
            base_cliques.append([u, v])
        best_cliques = base_cliques
        best_count = len(base_cliques)

        # Regime 2: Use I-vertices as hubs
        # A hub v with V_v ⊆ N_C[v] gives clique {v} ∪ V_v
        # Covers: all C-C edges within V_v, and all cross edges (u,v) for u in V_v
        # 
        # Strategy: greedily select hubs to cover C-C edges and cross edges
        # Each hub v can cover:
        #   - C-C edges: all pairs within N_C[v]
        #   - Cross edges: (u,v) for u in N_C[v]
        #
        # We want to minimize total number of cliques.
        # If we use hub v with full N_C[v], it covers all cross edges from v 
        # and all C-C edges among N_C[v].
        #
        # Key: we need to cover ALL C-C edges and ALL cross edges.
        # 
        # Approach: 
        # 1. For each I-vertex, compute what it can cover
        # 2. Greedily select hubs to cover C-C edges first (most expensive to cover otherwise)
        # 3. For remaining cross edges, either use additional hubs or 2-cliques

        # Precompute: for each v, the set of C-C edges it can cover
        v_cc_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cc_cover[v] = frozenset((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cc_cover[v] = frozenset()

        # Greedy set cover for C-C edges using I-vertices
        uncovered_cc = set(cc_edges)
        selected_hubs = []
        
        while uncovered_cc:
            best_v = None
            best_cnt = 0
            for v in range(k, n):
                if v in selected_hubs:
                    continue
                cnt = len(uncovered_cc & v_cc_cover[v])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None or best_cnt == 0:
                break
            selected_hubs.append(best_v)
            uncovered_cc -= v_cc_cover[best_v]

        # If we couldn't cover all C-C edges with hubs, fall back
        if uncovered_cc:
            # Use the baseline or try to cover remaining with 2-cliques
            # Actually, if some C-C edge can't be covered by any single hub,
            # we need the big C-clique or multiple smaller cliques
            # For now, use baseline if hub approach doesn't help
            return best_cliques

        # Now build cliques from selected hubs
        # Each hub v gets clique {v} ∪ N_C[v]
        # But we need to verify all cross edges are covered
        # Cross edge (u,v) is covered if v is a hub and u in N_C[v]
        
        hub_set = set(selected_hubs)
        cliques = []
        
        # Add hub cliques
        for v in selected_hubs:
            nc = sorted(N_C[v])
            if nc:  # non-empty
                clique = [v] + nc
                cliques.append(clique)
        
        # Check which cross edges are NOT covered by hubs
        # Cross edge (u,v) covered if v in hub_set and u in N_C[v]
        uncovered_cross = set()
        for (u, v) in cross_edges:
            if v not in hub_set or u not in N_C[v]:
                uncovered_cross.add((u, v))
        
        # For uncovered cross edges, add 2-cliques
        for (u, v) in sorted(uncovered_cross):
            cliques.append([u, v])
        
        # Verify: every edge is in exactly one clique
        # This is tricky - we need to ensure no edge appears in two cliques
        # 
        # Problem: if hub v has N_C[v] = {a, b, c}, clique {v,a,b,c} covers 
        # edges (a,b), (a,c), (b,c) in C and (a,v), (b,v), (c,v) cross.
        # But if another hub w also has a in N_C[w], then edge (a,?) might be double-covered.
        # 
        # Actually, C-C edges: edge (a,b) is covered by hub v if a,b in N_C[v].
        # It could also be covered by hub w if a,b in N_C[w].
        # We need each edge in EXACTLY one clique.
        #
        # This is the hard part. The greedy set cover might select overlapping hubs.
        # We need to ensure disjoint coverage.
        
        # Let me reconsider: the constraint is that every edge lies in exactly one clique.
        # This means the cliques must form an edge partition.
        
        # For the hub approach to work cleanly:
        # - Each C-C edge must be assigned to exactly one hub
        # - Each cross edge must be assigned to exactly one clique (hub or 2-clique)
        
        # Let me try a different approach: 
        # Partition C-C edges among hubs, and for each hub, V_v is the set of C-vertices
        # such that all C-C edges "assigned" to v have both endpoints in V_v,
        # and all cross edges (u,v) have u in V_v.
        
        # This is getting complex. Let me try a simpler structural change:
        # Use the C-clique for C-C edges, and for cross edges, group them by 
        # I-vertex. For each I-vertex v, if it has cross edges, create one clique 
        # {v} ∪ {all u with (u,v) cross edge}. This covers all cross edges from v 
        # in one clique.
        
        # Total cliques: 1 (for C) + number of I-vertices with at least one cross edge
        # This is much better than 1 + |cross_edges|
        
        # But wait: does {v} ∪ N_C[v] form a clique? 
        # v is connected to all u in N_C[v] by definition.
        # The u's are in C, which is a clique.
        # So yes, {v} ∪ N_C[v] is a clique.
        
        # And this covers all cross edges from v, plus C-C edges among N_C[v].
        # But C-C edges among N_C[v] are also covered by the big C-clique!
        # That's double coverage - violates "exactly one" rule.
        
        # So we can't use both the big C-clique and hub cliques that overlap in C-C edges.
        
        # Option A: Use big C-clique for all C-C edges, and for cross edges,
        # use per-I-vertex cliques but only include the cross edges, not C-C edges.
        # But a clique {v, u1, u2} contains edge (u1,u2) which is a C-C edge.
        # So we can't have {v, u1, u2} if (u1,u2) is already in the C-clique.
        
        # Unless... we remove those C-C edges from the C-clique? But the C-clique
        # must be a clique, so it contains all edges among its vertices.
        
        # Key insight: if we use clique {v} ∪ S where S ⊆ C, then all C-C edges
        # within S are "used up" by this clique. We can't also have them in another clique.
        
        # So the approach must be:
        # 1. Partition C-C edges among various cliques
        # 2. Each clique is either a subset of C, or {v} ∪ S_v where S_v ⊆ C
        # 3. No C-C edge appears in two cliques
        
        # This is essentially a clique partition problem.
        
        # For split graphs, a natural approach:
        # - The C-clique covers all C-C edges
        # - For cross edges, we need cliques that include cross edges but no C-C edges
        #   The only way: 2-cliques {u, v} for each cross edge
        #   OR: if we break up the C-clique, we can have larger cross cliques
        
        # Let me try: don't use the full C-clique. Instead, partition C into groups,
        # and for each I-vertex v, assign it to one group, forming {v} ∪ group.
        # But this requires the groups to partition C, and each cross edge (u,v) 
        # requires u and v to be in the same clique.
        
        # Actually, the simplest valid improvement: 
        # Keep the C-clique. For cross edges, use 2-cliques. That's the baseline.
        # 
        # To improve, we need to avoid double-covering C-C edges.
        # 
        # New idea: What if we DON'T use the full C-clique? 
        # Instead, for each I-vertex v, create clique {v} ∪ N_C[v].
        # This covers all cross edges and some C-C edges.
        # Remaining C-C edges (those not in any N_C[v] for any v) need separate cliques.
        # 
        # But C-C edge (a,b) is covered by hub v if a,b ∈ N_C[v].
        # If for some (a,b), no v has both a,b in N_C[v], then (a,b) isn't covered.
        # With p=0.7, most pairs should be covered by some v.
        
        # Let me implement this carefully:
        
        # Step 1: For each I-vertex v, define hub clique H_v = {v} ∪ N_C[v]
        # Step 2: Determine which C-C edges are covered by at least one H_v
        # Step 3: For uncovered C-C edges, add additional cliques
        # Step 4: Ensure no C-C edge is in two H_v's (if it is, we have a problem)
        
        # The problem: if (a,b) is in both H_v and H_w for v≠w, that's double coverage.
        # To fix: assign each C-C edge to exactly one hub.
        
        # This is a complex assignment problem. Let me try a pragmatic approach:
        # 
        # For each I-vertex v, let S_v = N_C[v]. The hub clique is {v} ∪ S_v.
        # C-C edges covered by v: all pairs in S_v.
        # Cross edges covered by v: all (u,v) for u in S_v.
        #
        # If we use ALL hubs, we might double-cover C-C edges.
        # 
        # Alternative: only use hubs for I-vertices, and handle C-C edges separately.
        # But as noted, {v} ∪ S_v contains C-C edges within S_v.
        
        # I think the cleanest approach given the constraints is:
        # 1. Use the full C-clique for all C-C edges
        # 2. For cross edges, use 2-cliques (can't do better without breaking C-clique)
        # 
        # Unless we can find a way to merge some 2-cliques into larger ones
        # without creating double C-C coverage.
        
        # Wait - what if two I-vertices v1, v2 share no common C-neighbor?
        # Then {v1, v2, u} where u is common neighbor... no, if no common neighbor,
        # we can't form {v1, v2, u}.
        
        # What about {v1, v2}? Is that a clique? v1 and v2 are both in I, 
        # and I is an independent set, so NO. v1-v2 is not an edge.
        
        # So we can't merge cross edges from different I-vertices.
        
        # The baseline (1 + |cross_edges|) might be close to optimal for this structure.
        # 
        # Let me check: can we do better than 1 + |cross_edges|?
        # 
        # Total edges = C(C-1)/2 + |cross_edges|
        # Each clique of size s covers C(s,2) edges.
        # To minimize number of cliques, we want large cliques.
        # 
        # The C-clique of size k covers C(k,2) edges in 1 clique.
        # Each 2-clique covers 1 edge.
        # Total: 1 + |cross_edges|
        #
        # Could we do better? Only if we can cover multiple cross edges in one clique.
        # A clique containing a cross edge (u,v) must contain u and v.
        # If it contains another cross edge (u',v'), it must contain u' and v'.
        # If v ≠ v', then we need edge v-v', but I is independent, so no.
        # So a clique can contain cross edges only from one I-vertex.
        # 
        # For I-vertex v, a clique {v, u1, u2, ..., um} covers cross edges (u1,v),...,(um,v)
        # and C-C edges among {u1,...,um}.
        # 
        # If we use such a clique, those C-C edges are "used up".
        # 
        # So the trade-off: using {v} ∪ N_C[v] saves (|N_C[v]| - 1) cliques 
        # (instead of |N_C[v]| 2-cliques) but uses up C(|N_C[v]|, 2) C-C edges.
        # 
        # Net benefit: we save |N_C[v]| - 1 cliques for cross edges, 
        # but we lose the ability to use those C-C edges in the C-clique.
        # 
        # If we remove those C-C edges from the C-clique, we need to cover them elsewhere.
        # But they're already covered by the hub clique!
        # 
        # So: if we use hub {v} ∪ N_C[v], we cover:
        # - |N_C[v]| cross edges (saving |N_C[v]| - 1 cliques vs 2-cliques)
        # - C(|N_C[v]|, 2) C-C edges (which were in the C-clique)
        # 
        # But now the C-clique can't include all of C, because some C-C edges are "used".
        # 
        # This is getting very complex. Let me just go with the baseline and see.
        
        return best_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() - start_time > DEADLINE - 5:
            break
        results.append(solve_one(*spec))
    
    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        results.append([[list(e)] for e in sorted(edges)])
    
    return results
